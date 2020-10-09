import os
import shutil
import time
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
                    type=str, choices=['video', 'is', 'npz'], default='npz',
                    help='Output file type, -o needs to be a file and image sequence or npz needs to be a folder')
# Process type
parser.add_argument('-a', '--algorithm', type=str, default='SSM',  # 算法
                    choices=['DAIN', 'SSM'], help='DAIN or SSM')
parser.add_argument('-pt', '--process_type',  # 如何处理
                    type=str, choices=['general', '60fps'], default='general',
                    help='1. General processing; 2. Interpolate to at least 60fps, ex. 30->60, 24->72， 50->100; Interpolate between duplicated frames. ')
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
                    help='specify start frame')
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
# Other
parser.add_argument('-mc', '--mac_compatibility',  # 让苹果设备可以直接播放
                    type=bool, default=True,
                    help='If you want to play it on a mac with QuickTime or iOS, set this to True and the pixel format will be yuv420p. ')
parser.add_argument('-bs', '--batch_size',  # Batch Size
                    type=int, default=1,
                    help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
# Temporary files
parser.add_argument('-tmp', '--temp_file_path',  # 临时文件路径
                    type=str, default='tmp',
                    help='Specify temporary file path, put it in your drive to make sure you don\'t lost them when colab disconnects. ')
parser.add_argument('-rm', '--remove_temp_file',  # 是否移除临时文件
                    type=bool, default=False,
                    help='If you want to keep temporary files, select True ')

# DAIN
parser.add_argument('-net', '--net_name', type=str, default='DAIN_slowmotion',  # DAIN 的网络
                    choices=['DAIN', 'DAIN_slowmotion'], help='model architecture: DAIN | DAIN_slowmotion')
parser.add_argument('-sw', '--save_which', type=int, default=1,  # 保存哪个
                    choices=[0, 1], help='choose which result to save: 0 ==> interpolated, 1==> rectified')

args = parser.parse_args()


def listdir(folder):  # 输入文件夹路径，输出文件夹内的文件，排序并移除可能的无关文件
    disallow = ['.DS_Store', '.ipynb_checkpoints', '$RECYCLE.BIN', 'Thumbs.db', 'desktop.ini']
    files = os.listdir(folder)
    for file in files:
        if file in disallow:
            files.remove(file)
    files.sort()
    return files


def detect_input_type(input_dir):  # 检测输入类型
    if os.path.isfile(input_dir):
        if os.path.splitext(input_dir)[1].lower() == '.txt':
            input_type = 'continue'
        else:
            input_type = 'video'
    else:
        files = listdir(input_dir)
        if os.path.splitext(files[0])[1].lower() == '.npz':
            input_type = 'npz'
        elif os.path.splitext(files[0])[1].replace('.', '').lower() in ['dpx', 'jpg', 'jpeg', 'exr', 'psd', 'png',
                                                                        'tif', 'tiff']:
            input_type = 'is'
        else:
            input_type = 'mix'
    return input_type


def split_file_name(file_path):
    file_path = list(os.path.split(file_path))
    file_path.extend(os.path.splitext(file_path[1]))
    file_path.pop(1)
    return file_path


def create_temp_folder(temp_file_path, filename):
    while True:
        temp_folder = os.path.join(temp_file_path, f'{filename}-{random.randint(1000000, 9999999)}')
        temp_folder = os.path.abspath(temp_folder)
        if not os.path.exists(temp_folder):
            break
    os.makedirs(temp_folder)
    os.makedirs(f'{temp_folder}/out')
    print(f'Created temporary folder: \n{temp_folder}')
    return temp_folder


def video_pre_process(video_path, current_temp_file_path, sf, fps, process_type):
    os.mkdir(f'{current_temp_file_path}/in')
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(7))
    frame_count_len = len(str(frame_count))
    original_fps = cap.get(5)

    count = 1
    while True:
        rtl, frame = cap.read()
        if rtl:
            numpy.savez_compressed(f'{current_temp_file_path}/in/{str(count).zfill(frame_count_len)}', frame)
        else:
            break
        count += 1

    cap.release()
    # sf
    if process_type == 'general':
        sf = sf
    elif process_type == '60fps':
        sf = math.ceil(60 / original_fps)

    # fps
    if fps:
        target_fps = fps
    else:
        target_fps = original_fps * sf
    # fix
    return {'frame_count': frame_count, 'frame_count_len': frame_count_len, 'original_fps': original_fps,
            'target_fps': target_fps, 'sf': sf}


def is_pre_process(img_sequence_path, current_temp_file_path, fps, sf):
    os.mkdir(f'{current_temp_file_path}/in')
    files = listdir(img_sequence_path)
    frame_count = len(files)
    frame_count_len = len(str(frame_count))
    for file in files:
        frame = cv2.imread(f'{img_sequence_path}/{file}')
        numpy.savez_compressed(f'{current_temp_file_path}/in/{os.path.splitext(file)[0]}', frame)
    if fps:
        target_fps = fps
    else:
        target_fps = 60
    return {'frame_count': frame_count, 'frame_count_len': frame_count_len, 'target_fps': target_fps, 'sf': sf}


def npz_pre_process(npz_path, current_temp_file_path, fps, sf):
    shutil.copytree(npz_path, f'{current_temp_file_path}/in')
    frame_count = len(listdir(npz_path))
    frame_count_len = len(str(frame_count))
    if fps:
        target_fps = fps
    else:
        target_fps = 60
    return {'frame_count': frame_count, 'frame_count_len': frame_count_len, 'target_fps': target_fps, 'sf': sf}


def pre_process(cag_, args_):
    if cag_['input_type'] == 'video':
        cag_add_ = video_pre_process(cag_['input_file_path'], cag_['current_temp_file_path'], args_['sf'], args_['fps'],
                                     args_['process_type'])
    if cag_['input_type'] == 'is':
        cag_add_ = is_pre_process(cag_['input_file_path'], cag_['current_temp_file_path'], args_['fps'], args_['sf'])
    if cag_['input_type'] == 'npz':
        cag_add_ = npz_pre_process(cag_['input_file_path'], cag_['current_temp_file_path'], args_['fps'], args_['sf'])
    return dict(cag_, **cag_add_)  # Merge dictionaries.


def process_start_end_frame(in_start_frame, in_end_frame, frame_count):
    if in_end_frame == 0 or in_end_frame == frame_count or in_end_frame > frame_count:
        copy = True
        end_frame = frame_count
    else:
        copy = False
        end_frame = in_end_frame
    if in_start_frame == 0 or in_start_frame >= frame_count:
        start_frame = 1
    else:
        start_frame = in_start_frame
    return start_frame, end_frame, copy


def read_cag(cag_path):
    with open(cag_path, 'r') as f_:
        cag_ = eval(f_.read())
    processed_frame_count_ = len(listdir(f'{cag_["current_temp_file_path"]}/out'))
    start_frame_ = int(processed_frame_count_ // cag['sf'])
    cag_['frames_to_process'] = cag_['original_frames_to_process'][start_frame_:]
    return cag_


def check_output_dir(set_output, filename, sf, output_type):
    # Check if it's set or by default
    if set_output == 'default':
        # If didn't specify output
        output_dir_ = f'{filename[0]}/{filename[1]}_{sf}x'
    else:
        output_dir_ = set_output
    # Check output name
    if output_type == 'video':
        if set_output == 'default':
            if filename[2]:
                ext = filename[2]
            else:
                ext = '.mp4'
        else:
            ext = os.path.splitext(set_output)[1]
    else:  # If output is npz or image_sequence
        if os.path.splitext(output_dir_)[1]:  # If output should be folder but set a extension
            output_dir_ = output_dir_.replace(os.path.splitext(output_dir_)[1], '')
        ext = ''

    # Check if output directory exists
    count = 2
    if os.path.exists(output_dir_ + ext):
        while True:
            if os.path.exists(f'{output_dir_}_{count}{ext}'):
                pass
            else:
                output_dir_ = f'{output_dir_}_{count}{ext}'
                break
            count += 1
    else:
        output_dir_ = output_dir_ + ext
    return output_dir_


def npz2tif(npz_path, tiff_path):  # 把npz转成tiff
    os.makedirs(tiff_path)
    frames_to_process = listdir(npz_path)
    for frame in frames_to_process:
        img = numpy.load(f'{npz_path}/{frame}')['arr_0']
        cv2.imwrite(f'{tiff_path}/{frame.split(".")[0]}.tiff', img)


# Process Info
args = args.__dict__
# 算法
if args['algorithm'] == 'SSM':
    from SSM.interpolate import main
elif args['algorithm'] == 'DAIN':
    from DAIN.interpolate import main
else:
    def main(**error):
        print(error)
        exit(1)

# Model checking
model_path = {'DAIN': 'DAIN/model_weights/best.pth', 'SSM': 'SSM/SuperSloMo.ckpt'}
if args['model_path'] == 'default':  # 模型路径
    args['model_path'] = model_path[args['algorithm']]

if not os.path.exists(args["model_path"]):
    print(f"Model {args['algorithm']}/{args['model_path']} doesn't exist, exiting")
    exit(1)

input_type = detect_input_type(args['input'])  # 把要处理的一个或多个视频放入一个列表
if input_type == 'mix':
    processes = listdir(args['input'])
    processes = [os.path.join(args['input'], process) for process in processes]
else:
    processes = [args['input']]

for process in processes:
    cag = {'model_path': args['model_path'],
           'batch_size': args['batch_size'],
           'start_frame': args['start_frame'],
           'output': args['output'],
           'output_type': args['output_type'],
           'mac_compatibility': args['mac_compatibility'],
           'remove_temp_file': args['remove_temp_file'],
           'ffmpeg_dir': args['ffmpeg_dir'],
           'vcodec': args['vcodec']}  # Current arguments
    if process != 'continue':  # If not continue
        cag['input_file_path'] = process
        cag['input_file_name_list'] = split_file_name(cag['input_file_path'])
        cag['current_temp_file_path'] = create_temp_folder(args['temp_file_path'], cag['input_file_name_list'][1])
        # Detect input type and process it.
        cag['input_type'] = detect_input_type(cag['input_file_path'])
        cag = pre_process(cag, args)
        # Start/End frame
        cag['start_frame'], cag['end_frame'], cag['copy'] = process_start_end_frame(args['start_frame'], args['end_frame'], cag['frame_count'])
        # Frames to process
        cag['original_frames_to_process'] = listdir(f'{cag["current_temp_file_path"]}/in')[
                                            cag['start_frame'] - 1:cag['end_frame']]
        cag['frames_to_process'] = cag['original_frames_to_process']
    else:
        cag = read_cag(process)

    # Log
    with open(f'{cag["current_temp_file_path"]}/process_info.txt', 'w') as f:
        f.write(str(cag))

    # Process
    t = time.time()
    main(f'{cag["current_temp_file_path"]}/process_info.txt')
    print(f'Interpolation spent {round(time.time() - t, 2)}s')

    # if cag['copy']:
    #     # Copy last frame
    #     original = f'{cag["current_temp_file_path"]}/in/{cag["original_frames_to_process"][-1]}'
    #     for i in range(cag['sf']):
    #         target = f'{cag["current_temp_file_path"]}/out/{cag["original_frames_to_process"][-1].replace(".npz", "")}' \
    #                  f'_{str(i).zfill(len(str(cag["sf"] - 1)))}.npz'
    #         shutil.copyfile(original, target)

    output_dir = check_output_dir(cag['output'], cag["input_file_name_list"], cag["sf"], cag['output_type'])

    if cag['output_type'] == 'video':
        # Mac compatibility
        pix_fmt = ' -pix_fmt yuv420p' if cag['mac_compatibility'] else ''
        npz2tif(f"{cag['current_temp_file_path']}/out", f"{cag['current_temp_file_path']}/tiff")
        # Check file extension
        if cag['input_type'] == 'video' and cag['start_frame'] == 1 and cag['end_frame'] == 0:
            cmd = f'"{os.path.join(cag["ffmpeg_dir"], "ffmpeg")}" -loglevel error ' \
                  f'-thread_queue_size 128 ' \
                  f'-vsync 0 -r {cag["target_fps"]} -pattern_type glob -i "{cag["current_temp_file_path"]}/tiff/*.tiff" ' \
                  f'-vn -i "{cag["temp_folder"]}/in{cag["filename"][2]}" ' \
                  f'-vcodec {cag["vcodec"]}{pix_fmt} -b 100M "{output_dir}"'
        else:
            cmd = f'"{os.path.join(cag["ffmpeg_dir"], "ffmpeg")}" -loglevel error ' \
                  f'-vsync 0 -r {cag["target_fps"]} -pattern_type glob -i "{cag["current_temp_file_path"]}/tiff/*.tiff" ' \
                  f'-vcodec {cag["vcodec"]}{pix_fmt} -b 100M "{output_dir}"'
        print(cmd)
        os.system(cmd)
    if cag['output_type'] == 'is':
        npz2tif(f'{cag["current_temp_file_path"]}/out', output_dir)
    if cag['output_type'] == 'npz':
        shutil.copytree(f'{cag["current_temp_file_path"]}/out', output_dir)

    if cag['remove_temp_file']:
        shutil.rmtree(cag["current_temp_file_path"])
