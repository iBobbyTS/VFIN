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
import torch

parser = argparse.ArgumentParser()

# Input/output file
parser.add_argument('-i', '--input',  # Input file
                    type=str,
                    help='path of video to be converted')
# Process type
parser.add_argument('-a', '--algorithm', type=str, default='SSM',  # 算法
                    choices=['DAIN', 'SSM'], help='DAIN or SSM')
# Model directory
parser.add_argument('-md', '--model_path',  # 模型路径
                    type=str, default='default',
                    help='path of checkpoint for pretrained model')
# Time step
parser.add_argument('-sf',  # 多少倍帧率
                    type=int, default=2,
                    help='specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames')
# Other
parser.add_argument('-ec', '--empty_cache',  # Batch Size
                    type=int, default=0,
                    help='Empty cache while processing, set to 1 if you get CUDA out of memory errors; If there\'s '
                         'the process is ok, setting to 1 will slow down the process. ')

args = parser.parse_args()

model_path = {'DAIN': 'model_weights/best.pth', 'SSM': 'SuperSloMo.ckpt'}


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = os.listdir(folder)
    for file in files:
        if file in disallow:
            files.remove(file)
    files.sort()
    return files


class data_loader:
    def __init__(self, input_dir, input_type, sf):
        self.input_type = input_type
        self.input_dir = input_dir
        self.sf = sf
        if input_type == 'video':
            self.cap = cv2.VideoCapture(input_dir)
            self.fps = self.cap.get(5)
            self.frame_count = int(self.cap.get(7))
            self.height = self.cap.get(4)
            self.width = self.cap.get(3)
        else:
            self.count = -1
            self.files = listdir(input_dir)
            self.frame_count = len(self.files)
            self.img = cv2.imread(self.files[0]).shape
            self.height = self.img[0]
            self.height = self.img[1]
            del self.img
        self.read = self.video_func if self.input_type == 'video' else self.sequence_func

    def video_func(self):
        self.frame = self.cap.read()
        if self.frame[0]:
            return self.frame[1]

    def sequence_func(self):
        self.count += 1
        return {'is': cv2.imread,
                'npz': lambda path: numpy.load(path)['arr_0'],
                'npy': numpy.load
                }[self.input_type](f'{self.input_dir}/{self.files[self.count]}')

    def close(self):
        if self.input_type == 'video':
            self.cap.close()


def detect_input_type(input_dir):  # 检测输入类型
    if os.path.isfile(input_dir):
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


def second2time(second: float):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    t = '%d:%02d:%02d' % (h, m, s)
    return t


class mse:
    def __init__(self):
        self.final_loss = []
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

    def add(self, a: numpy.ndarray, b: numpy.ndarray):
        self.final_loss.append(float(self.loss_fn(
            torch.autograd.Variable(torch.FloatTensor(a.copy())),
            torch.autograd.Variable(torch.FloatTensor(b.copy()))
        )))

    def get_loss(self):
        return sum(self.final_loss) / len(self.final_loss)


args = args.__dict__

input_type = detect_input_type(args['input'])
if input_type == 'mix':
    processes = listdir(args['input'])
    processes = [os.path.join(args['input'], process) for process in processes]
else:
    processes = [args['input']]

for input_file_path in processes:
    input_type = detect_input_type(input_file_path)
    input_file_name_list = list(os.path.split(input_file_path))
    input_file_name_list.extend(os.path.splitext(input_file_name_list[1]))
    input_file_name_list.pop(1)
    cap = cv2.VideoCapture(input_file_path)
    sf = args['sf']
    frame_count = int(cap.get(7))
    frame_count_len = len(str(frame_count))
    # Setup
    if 'Interpolator' not in locals():
        if args['empty_cache']:
            os.environ['CUDA_EMPTY_CACHE'] = '1'
        if args['model_path'] == 'default':  # 模型路径
            model_path = f"{args['algorithm']}/{model_path[args['algorithm']]}"
        # Model checking
        if not os.path.exists(model_path):
            print(f"Model {model_path} doesn't exist, exiting")
            exit(1)
        sys.path.append(f"{os.path.abspath(args['algorithm'])}")
        from interpolator import Interpolator
    # Interpolate
    interpolator = Interpolator(model_path, sf, int(cap.get(4)), int(cap.get(3)), batch_size=1,
                                save_which=1, net_name='DAIN_slowmotion', channel=3)
    timer = 0
    loss_fn = mse()
    start_time = time.time()
    batch = [cap.read()[1]]
    batch_count = frame_count//sf-1
    for i in range(batch_count):
        gt = []
        for f in [cap.read() for _ in range(sf-1)]:
            if f[0]:
                gt.append(f[1])
            else:
                break
        f = cap.read()
        if f[0]:
            del batch[:-1]
            batch.append(f[1])
        intermediate_frames = interpolator.interpolate(batch)

        # Time
        time_spent = time.time() - start_time
        start_time = time.time()
        for t, p in zip(gt, [_[0] for _ in intermediate_frames]):
            loss_fn.add(t, p)
        if i == 0:
            initialize_time = time_spent
            print(f'Initialized and processed frame 0/{frame_count} | '
                  f'{batch_count - i - 1} frames left | '
                  f'Time spent: {round(initialize_time, 2)}s | '
                  f'Loss: {loss_fn.get_loss()}',
                  end='')
        else:
            timer += time_spent
            frames_processes = i + 1
            frames_left = batch_count - frames_processes
            print(f'\rProcessed frame {frames_processes}/{batch_count} | '
                  f"{frames_left} {'frames' if frames_left > 1 else 'frame'} left | "
                  f'Time spent: {round(time_spent, 2)}s | '
                  f'Time left: {second2time(frames_left * timer / i)} | '
                  f'Total time spend: {second2time(timer + initialize_time)} | '
                  f'Loss: {loss_fn.get_loss()}',
                  end='', flush=True)
    print(f"\rFilename: {''.join(input_file_name_list[1:3])} | Loss: {loss_fn.get_loss()} | Total time spend: {second2time(timer + initialize_time)}", flush=True)
