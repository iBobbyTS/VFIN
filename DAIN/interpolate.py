import time
import os
import shutil
import warnings
import numpy
import cv2
import torch
from torch.autograd import Variable
import networks as networks
from empty_cache import empty_cache


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = os.listdir(folder)
    for file in files:
        if file in disallow:
            files.remove(file)
    files.sort()
    return files


class data_loader(object):
    def __init__(self, input_dir, input_type, start_frame):
        self.input_type = input_type
        self.input_dir = input_dir
        self.start_frame = start_frame
        if input_type == 'video':
            self.cap = cv2.VideoCapture(input_dir)
            self.cap.set(1, self.start_frame)
            self.file_count = int(self.cap.get(7) - self.start_frame)
        else:
            self.count = -1
            self.files = listdir(input_dir)[self.start_frame:]
            self.file_count = len(self.files)

    def get(self):  # get frame
        if self.input_type == 'video':
            self.frame = self.cap.read()
            if self.frame[0]:
                return self.frame[1]
        else:
            self.count += 1
            self.frame_dir = f'{self.input_dir}/{self.files[self.count]}'
            if self.input_type == 'is':
                if os.path.exists(self.frame_dir):
                    return cv2.imread(self.frame_dir)
            if self.input_type == 'npz':
                if os.path.exists(self.frame_dir):
                    return numpy.load(self.frame_dir)['arr_0']


def second2time(second: float):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    time = '%d:%02d:%02d' % (h, m, s)
    return time


def main(process_info):
    start_time = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.backends.cudnn.benchmark = True
    
    sf_length = len(str(process_info['sf'] - 1))

    # Load data
    video = data_loader(process_info['input_file_path'], process_info['input_type'], process_info['interpolation_start_frame'])


    model = networks.__dict__[process_info['net_name']](
        channel=3,
        filter_size=4,
        timestep=1 / process_info['sf'],
        training=False).cuda()
    empty_cache()
    model_path = process_info['model_path']
    
    pretrained_dict = torch.load(model_path)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []

    model = model.eval()  # deploy mode

    timestep = 1 / process_info['sf']
    time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]

    torch.set_grad_enabled(False)
    
    # input_files = process_info['frames_to_process']
    frame_count = video.file_count - 1
    frame_count_len = len(str(frame_count + 1))
    loop_timer = []
    frames_left = frame_count - 1
    frames_processed = 1
    total_second_spent = 0
    try:
        X1_ori = torch.cuda.FloatTensor(video.get())[:, :, :3].permute(2, 0, 1) / 255
        empty_cache()
        for _ in range(process_info['interpolation_start_frame'], frame_count):
            X0 = X1_ori
            X1 = torch.cuda.FloatTensor(video.get())[:, :, :3].permute(2, 0, 1) / 255
            empty_cache()
            X1_ori = X1

            assert (X0.size() == X1.size())

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channels = X0.size(0)
            if not channels == 3:
                print(
                    f"Skipping frame {_}, expected 3 color channels but found {channels}.")
                continue

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft = int((intWidth_pad - intWidth) / 2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intPaddingLeft = 32
                intPaddingRight = 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            X0 = Variable(torch.unsqueeze(X0, 0))
            X1 = Variable(torch.unsqueeze(X1, 0))
            X0 = pader(X0)
            X1 = pader(X1)
            empty_cache()

            y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
            empty_cache()
            y_ = y_s[process_info['save_which']]

            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = [y_.data.cpu().numpy()]
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
            X0 = numpy.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                         intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
            y_ = [numpy.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                          intPaddingLeft:intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
            offset = [numpy.transpose(
                offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for offset_i in offset]
            filter = [numpy.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter] if filter is not None else None
            X1 = numpy.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                         intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
            empty_cache()
            interpolated_frame_number = 0
            numpy.savez_compressed(f"{process_info['current_temp_file_path']}/out/{str(_).zfill(frame_count_len)}_{'0'.zfill(sf_length)}", numpy.round(X0).astype('uint8'))
            for item, time_offset in zip(y_, time_offsets):
                interpolated_frame_number += 1
                numpy.savez_compressed(f'{process_info["current_temp_file_path"]}/out/{str(_).zfill(frame_count_len)}_{str(interpolated_frame_number).zfill(sf_length)}',
                                       numpy.round(item).astype('uint8'))

            time_spent = time.time() - start_time
            start_time = time.time()
            if _ - process_info['interpolation_start_frame'] == 0:
                initialize_time = time_spent
                print(f'Initialized model and processed frame 1 | '
                      f'{frame_count-1} frames left | '
                      f'Time spent: {round(time_spent, 2)}s', end='')
            else:
                total_second_spent += time_spent
                frames_left -= 1
                frames_processed += 1
                print(f'\rProcessed frame {frames_processed}/{frame_count} | '
                      f'{frames_left} frames left | '
                      f'Time spent: {round(time_spent, 2)}s | '
                      f'Time left: {second2time(round(frames_left * total_second_spent / _, 2))} | '
                      f'Total time spend: {second2time(total_second_spent + initialize_time)}', end='', flush=True)
    except KeyboardInterrupt:
        exit(1)
    # Copy
    if process_info['copy']:
        for i in range(1, process_info['sf']+1):
            numpy.savez_compressed(f'{process_info["current_temp_file_path"]}/out/{str(frame_count).zfill(frame_count_len)}_{str(i).zfill(sf_length)}', numpy.round(X1).astype('uint8'))
