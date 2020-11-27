import time

everything_start_time = time.time()

import os
import sys
import shutil
import json
import math
import random
import argparse

import cv2
import numpy

parser = argparse.ArgumentParser()

# Input/output file
parser.add_argument('-i', '--input',  # Input file
                    type=str,
                    help='path of video to be converted')
parser.add_argument('-o', '--output',  # Output file
                    type=str, default='default',
                    help='Specify output file name. Default: output.mp4')
parser.add_argument('-ot', '--output_type',  # Output file type
                    type=str, choices=['video', 'npz', 'npy', 'tiff', 'png'], default='npy',
                    help='Output file type, -o needs to be a file and image sequence or npz needs to be a folder')

# Process type
parser.add_argument('-a', '--algorithm', type=str, default='SSM',  # 算法
                    choices=['DAIN', 'SSM'], help='DAIN or SSM')
parser.add_argument('-pt', '--process_type',  # 如何处理
                    type=str, choices=['general', '60fps'], default='general',
                    help='1. General processing; '
                         '2. Interpolate to at least 60fps; '
                         '3. Interpolate between duplicated frames. ')
# Model directory
parser.add_argument('-md', '--model_path',  # 模型路径
                    type=str, default='default',
                    help='path of checkpoint for pretrained model')
# Time step
parser.add_argument('-sf',  # 多少倍帧率
                    type=int, default=2,
                    help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
parser.add_argument('-fps',  # 目标帧率
                    type=float,
                    help='specify fps of output video. Default: original fps * sf.')
# Start/End frame
parser.add_argument('-st', '--start_frame',  # 开始帧
                    type=int, default=1,
                    help='specify start frame (Start from 1)')
parser.add_argument('-ed', '--end_frame',  # 结束帧
                    type=int, default=0,
                    help='specify end frame. Default: Final frame')
# FFmpeg
parser.add_argument('-fd', '--ffmpeg_dir',  # FFmpeg路径
                    type=str, default='',
                    help='path to ffmpeg(.exe)')
parser.add_argument('-vc', '--vcodec',  # 视频编码
                    type=str, default='h264',
                    help='Video codec')
parser.add_argument('-br', '--bit_rate',  # 视频编码
                    type=str, default='100M',
                    help='Bit rate for output video')
# Other
parser.add_argument('-mc', '--mac_compatibility',  # 让苹果设备可以直接播放
                    type=bool, default=True,
                    help='If you want to play it on a mac with QuickTime or iOS, set this to True and the pixel '
                         'format will be yuv420p. ')
parser.add_argument('-bs', '--batch_size',  # Batch Size
                    type=int, default=1,
                    help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
parser.add_argument('-ec', '--empty_cache',  # Batch Size
                    type=int, default=0,
                    help='Empty cache while processing, set to 1 if you get CUDA out of memory errors; If there\'s '
                         'the process is ok, setting to 1 will slow down the process. ')
# Temporary files
parser.add_argument('-tmp', '--temp_file_path',  # 临时文件路径
                    type=str, default='tmp',
                    help='Specify temporary file path')
parser.add_argument('-rm', '--remove_temp_file',  # 是否移除临时文件
                    type=bool, default=False,
                    help='If you want to keep temporary files, select True ')

# DAIN
parser.add_argument('-net', '--net_name', type=str, default='DAIN_slowmotion',  # DAIN 的网络
                    choices=['DAIN', 'DAIN_slowmotion'], help='model architecture: DAIN | DAIN_slowmotion')

args = parser.parse_args()

model_paths = {'DAIN': 'model_weights/best.pth', 'SSM': 'SuperSloMo.ckpt'}


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = []
    for file in os.listdir(folder):
        if file not in disallow and file[:2] != '._':
            files.append(file)
    files.sort()
    return files


class data_loader:
    def __init__(self, input_dir, input_type, start_frame):
        self.input_type = input_type
        self.input_dir = input_dir
        self.start_frame = start_frame
        self.sequence_read_funcs = {'is': cv2.imread,
                                    'npz': lambda path: numpy.load(path)['arr_0'],
                                    'npy': numpy.load
                                    }
        self.read = self.video_func if self.input_type == 'video' else self.sequence_func
        if input_type == 'video':
            self.cap = cv2.VideoCapture(input_dir)
            self.cap.set(1, self.start_frame)
            self.fps = self.cap.get(5)
            self.frame_count = int(self.cap.get(7))
            self.height = int(self.cap.get(4))
            self.width = int(self.cap.get(3))

        else:
            self.count = -1
            self.files = [f'{input_dir}/{f}' for f in listdir(input_dir)[self.start_frame:]]
            self.frame_count = len(self.files)
            self.img = self.sequence_read_funcs[input_type](self.files[0]).shape
            self.height = self.img[0]
            self.width = self.img[1]
            del self.img

        self.read = self.video_func if self.input_type == 'video' else self.sequence_func

    def video_func(self):
        return self.cap.read()

    def sequence_func(self):
        self.count += 1
        if self.count < self.frame_count:
            img = self.sequence_read_funcs[self.input_type](self.files[self.count])
            if img is not None:
                return True, img
        return False, None

    def close(self):
        if self.input_type == 'video':
            self.cap.close()


def data_writer(output_type):
    return {'tiff': lambda path, img: cv2.imwrite(path + '.tiff', img),
            'png': lambda path, img: cv2.imwrite(path + '.png', img),
            'npz': numpy.savez_compressed,
            'npy': numpy.save
            }[output_type]


def detect_input_type(input_dir):  # 检测输入类型
    if os.path.isfile(input_dir):
        if os.path.splitext(input_dir)[1].lower() == '.json':
            input_type_ = 'continue'
        else:
            input_type_ = 'video'
    else:
        files = listdir(input_dir)
        if os.path.splitext(files[0])[1].lower() == '.npz':
            input_type_ = 'npz'
        elif os.path.splitext(files[0])[1].lower() == '.npy':
            input_type_ = 'npy'
        elif os.path.splitext(files[0])[1].replace('.', '').lower() in \
                ['dpx', 'jpg', 'jpeg', 'exr', 'psd', 'png', 'tif', 'tiff']:
            input_type_ = 'is'
        else:
            input_type_ = 'mix'
    return input_type_


def check_output_dir(dire, ext=''):
    if not os.path.exists(os.path.split(dire)[0]):  # If mother directory doesn't exist
        os.makedirs(os.path.split(dire)[0])  # Create one
    if os.path.exists(dire+ext):  # If target file/folder exists
        count = 2
        while os.path.exists(f'{dire}_{count}{ext}'):
            count += 1
        dire = f'{dire}_{count}{ext}'
    if not ext:  # Output as folder
        os.mkdir(dire)
    return dire


def second2time(second: float):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    t = '%d:%02d:%.2f' % (h, m, s)
    return t


args = args.__dict__

input_type = detect_input_type(args['input'])
if input_type == 'mix':
    processes = listdir(args['input'])
    processes = [os.path.join(args['input'], process) for process in processes]
else:
    processes = [args['input']]
# Extra work
args['start_frame'] -= 1
for input_file_path in processes:
    input_type = detect_input_type(input_file_path)
    if input_type != 'continue':
        input_file_name_list = list(os.path.split(input_file_path))
        input_file_name_list.extend(os.path.splitext(input_file_name_list[1]))
        input_file_name_list.pop(1)
        temp_file_path = check_output_dir(os.path.join(args['temp_file_path'], input_file_name_list[1]))
        video = data_loader(input_file_path, input_type, args['start_frame'])
        frame_count = video.frame_count
        frame_count_len = len(str(frame_count))
        sf = args['sf']
        if args['fps']:
            original_fps = args['fps']
        elif input_type == 'video':
            original_fps = video.fps
        else:
            original_fps = 30
        target_fps = original_fps * sf

        if args['end_frame'] == 0 or args['end_frame'] == frame_count or args['end_frame'] > frame_count:
            copy = True
            end_frame = frame_count
        else:
            copy = False
            end_frame = args['end_frame'] + 1
        if args['start_frame'] == 0 or args['start_frame'] >= frame_count:
            start_frame = 1
        else:
            start_frame = args['start_frame']

        if args['model_path'] == 'default':  # 模型路径
            model_path = f"{args['algorithm']}/{model_paths[args['algorithm']]}"
        else:
            model_path = args['model_path']

        output_type = args['output_type']
        output_dir = args['output']
        if output_dir == 'default':
            output_dir = f"{input_file_name_list[0]}/{input_file_name_list[1]}{args['algorithm']}"
        if output_type == 'video':
            if input_file_name_list[2]:
                ext = input_file_name_list[2]
            else:
                ext = '.mp4'
        else:
            output_dir, ext = os.path.splitext(output_dir)
        output_dir = check_output_dir(output_dir, ext)
        if output_type == 'video':
            dest_path = check_output_dir(os.path.splitext(output_dir)[0], ext)
            output_dir = f'{temp_file_path}/tiff'
            output_type = 'tiff'
            os.makedirs(output_dir)
        else:
            dest_path = False

        cag = {'input_file_path': input_file_path,
               'input_type': input_type,
               'empty_cache': args['empty_cache'],
               'model_path': model_path,
               'temp_folder': temp_file_path,
               'algorithm': args['algorithm'],
               'sf': sf,
               'sf_len': len(str(sf)),
               'frame_count': frame_count,
               'frame_count_len': len(str(video.frame_count)),
               'height': video.height,
               'width': video.width,
               'start_frame': start_frame,
               'end_frame': end_frame,
               'batch_size': args['batch_size'],
               'net_name': args['net_name'],
               'output_type': output_type,
               'output_dir': output_dir,
               'dest_path': dest_path,
               'copy': copy,
               'mac_compatibility': args['mac_compatibility'],
               'ffmpeg_dir': args['ffmpeg_dir'],
               'target_fps': target_fps,
               'vcodec': args['vcodec']
               }

        with open(f'{temp_file_path}/process_info.json', 'w') as f:
            json.dump(cag, f)
    else:
        with open(input_file_path, 'r') as f_:
            cag = json.load(f_)
        start_frame = len(listdir(cag['output_dir'])) // cag['sf']
        video = data_loader(cag['input_file_path'], cag['input_type'], start_frame - 1)

    if cag['empty_cache']:
        os.environ['CUDA_EMPTY_CACHE'] = '1'

    # Model checking
    if not os.path.exists(cag['model_path']):
        print(f"Model {cag['model_path']} doesn't exist, exiting")
        exit(1)
    # Start frame
    batch_count = (cag['frame_count'] - start_frame) // cag['batch_size']
    if (cag['frame_count'] - start_frame - 1) % cag['batch_size']:
        batch_count += 1

    # Setup
    sys.path.append(f"{os.path.abspath(cag['algorithm'])}")
    from interpolator import Interpolator

    # Interpolate
    interpolator = Interpolator(cag['model_path'], cag['sf'], int(cag['height']), int(cag['width']), batch_size=cag['batch_size'],
                                net_name=cag['net_name'], channel=3  # DAIN
                               )
    save = data_writer(cag['output_type'])
    ori_frames = []
    batch = [None] * (cag['batch_size'])
    batch.append(video.read()[1])
    interpolator.batch[0] = interpolator.ndarray2tensor([batch[-1]])[0]
    timer = 0
    start_time = time.time()
    try:
        for i in range(batch_count):
            batch[0] = batch[-1]
            batch[1:] = [video.read() for _ in range(cag['batch_size'])]
            batch[1:] = [frame[1] for frame in batch[1:] if frame[0]]
            intermediate_frames = interpolator.interpolate(batch[1:])
            for frame_index, interpolated_frame in enumerate(batch[:-1]):
                save(f"{cag['output_dir']}/"
                     f"{str(i * cag['batch_size'] + frame_index + start_frame - 1).zfill(cag['frame_count_len'])}_"
                     f"{'0'.zfill(cag['sf_len'])}", batch[frame_index])
            for frame_index, interpolated_batch in enumerate(intermediate_frames):
                for batch_index, interpolated_frame in enumerate(interpolated_batch):
                    save(f"{cag['output_dir']}/"
                         f"{str(i * cag['batch_size'] + batch_index + start_frame - 1).zfill(cag['frame_count_len'])}_"
                         f"{str(frame_index + 1).zfill(cag['sf_len'])}", interpolated_frame)
            # Time
            time_spent = time.time() - start_time
            start_time = time.time()
            if i == 0:
                initialize_time = time_spent
                print(f'Initialized and processed frame 1/{batch_count} | '
                      f'{batch_count - i - 1} frames left | '
                      f'Time spent: {round(initialize_time, 2)}s',
                      end='')
            else:
                timer += time_spent
                frames_processes = i + 1
                frames_left = batch_count - frames_processes
                print(f'\rProcessed batch {frames_processes}/{batch_count} | '
                      f"{frames_left} {'batches' if frames_left > 1 else 'batch'} left | "
                      f'Time spent: {round(time_spent, 2)}s | '
                      f'Time left: {second2time(frames_left * timer / i)} | '
                      f'Total time spend: {second2time(timer + initialize_time)}', end='', flush=True)

    except KeyboardInterrupt:
        print('\nCaught Ctrl-C, exiting. ')
        exit(256)
    if cag['copy']:
        for i in range(cag['sf']):
            save(f"{cag['output_dir']}/"
                 f"{str(frame_count - 1).zfill(cag['frame_count_len'])}_"
                 f"{str(i).zfill(cag['sf_len'])}", batch[-1])
    del batch, interpolator
    print(f'\r{os.path.split(input_file_path)[1]} done! Total time spend: {second2time(timer + initialize_time)}',
          flush=True)

    # Post process
    if cag['dest_path']:
        # Mac compatibility
        pix_fmt = ' -pix_fmt yuv420p' if cag['mac_compatibility'] else ''
        # Execute command
        cmd = [f"'{os.path.join(cag['ffmpeg_dir'], 'ffmpeg')}' -loglevel error ",
               f"-vsync 0 -r {cag['target_fps']} -pattern_type glob -i '{cag['temp_folder']}/tiff/*.tiff' ",
               f"-vcodec {cag['vcodec']}{pix_fmt} '{cag['dest_path']}'"]
        if cag['start_frame'] == 1 and cag['end_frame'] == 0:
            cmd.insert(1, '-thread_queue_size 128 ')
            cmd.insert(3, f"-vn -i '{input_file_path}' ")
        cmd = ''.join(cmd)
        print(cmd)
        os.system(cmd)
print(time.time() - everything_start_time)
