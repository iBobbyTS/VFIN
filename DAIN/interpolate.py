import time
import os
from torch.autograd import Variable
import torch
import numpy
import networks as networks
import shutil
import warnings


def main(process_info):
    start_time = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.backends.cudnn.benchmark = True
    
    process_info = eval(process_info)
    sf_length = len(str(process_info['sf'] - 1))

    model = networks.__dict__[process_info['net_name']](
        channel=3,
        filter_size=4,
        timestep=1 / process_info['sf'],
        training=False).cuda()
    torch.cuda.empty_cache()
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
    
    input_files = process_info['frames_to_process']
    loop_timer = []
    try:
        X1_ori = torch.cuda.FloatTensor(numpy.load(f'{process_info["current_temp_file_path"]}/in/{input_files[0]}')['arr_0'])[:, :, :3].permute(2, 0, 1) / 255
        torch.cuda.empty_cache()
        for _ in range(len(input_files) - 1):
            filename_frame_2 = f'{process_info["current_temp_file_path"]}/in/{input_files[_ + 1]}'

            X0 = X1_ori
            X1 = torch.cuda.FloatTensor(numpy.load(filename_frame_2)['arr_0'])[:, :, :3].permute(2, 0, 1) / 255
            torch.cuda.empty_cache()
            X1_ori = X1

            assert (X0.size() == X1.size())

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channels = X0.size(0)
            if not channels == 3:
                print(
                    f"Skipping {filename_frame_2} -- expected 3 color channels but found {channels}.")
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
#             torch.cuda.empty_cache()
            X1 = pader(X1)
            torch.cuda.empty_cache()

            y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
            torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
            
            interpolated_frame_number = 0
            shutil.copy(f'{process_info["current_temp_file_path"]}/in/{input_files[_]}',
                        f'{process_info["current_temp_file_path"]}/out/{input_files[_].replace(".npz", "")}_{"0".zfill(sf_length)}.npz')
            for item, time_offset in zip(y_, time_offsets):
                interpolated_frame_number += 1
                output_frame_file_path = f'{process_info["current_temp_file_path"]}/out/{input_files[_].replace(".npz", "")}_{str(interpolated_frame_number).zfill(sf_length)}'
                numpy.savez_compressed(output_frame_file_path, numpy.round(item).astype('uint8'))

            time_spent = time.time() - start_time
            if process_info['reinitialize'] == 0:
                if _ == 0:
                    print(f"Initialized model and processed frame {process_info['frames_to_process'][_ + 1].split('.')[0]} | Time spent: {round(time_spent, 2)}s", end='')
                else:
                    loop_timer.append(time_spent)
                    frames_left = len(input_files) - _ - 2
                    estimated_seconds_left = round(frames_left * sum(loop_timer) / len(loop_timer), 2)
                    m, s = divmod(estimated_seconds_left, 60)
                    h, m = divmod(m, 60)
                    estimated_time_left = "%d:%02d:%02d" % (h, m, s)
                    print(f"\rProcessed frame {process_info['frames_to_process'][_ + 1].split('.')[0]} | Time spent: {round(time_spent, 2)}s | Time left: {estimated_time_left}", end='', flush=True)
            else:
                print(f"\rProcessed frame {process_info['frames_to_process'][0].split('.')[0]} | Time spent: {round(time_spent, 2)}s", end='', flush=True)
            start_time = time.time()

    except KeyboardInterrupt:
        exit(1)
