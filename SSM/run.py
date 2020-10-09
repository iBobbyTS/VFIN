import os
import sys
import shutil
import time
import random
import argparse
import cv2
import numpy


def str2bool(v):  # Convert string to bool
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True


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
# Other
parser.add_argument('-fd', '--ffmpeg_dir',  # FFmpeg路径
                    type=str, default='',
                    help='path to ffmpeg(.exe)')
parser.add_argument('-mc', '--mac_compatibility',  # 让苹果设备可以直接播放
                    type=str2bool, default=True,
                    help='If you want to play it on a mac with QuickTime or iOS, set this to True and the pixel format will be yuv420p. ')
parser.add_argument('-bs', '--batch_size',  # Batch Size
                    type=int, default=1,
                    help='Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1')
# Temporary files
parser.add_argument('-tmp', '--temp_file_path',  # 临时文件路径
                    type=str, default='tmp',
                    help='Specify temporary file path, put it in your drive to make sure you don\'t lost them when colab disconnects. ')
parser.add_argument('-rm', '--remove_temp_file',  # 是否移除临时文件
                    type=str2bool, default=False,
                    help='If you want to keep temporary files, select True ')
# DAIN
parser.add_argument('-net', '--net_name', type=str, default='DAIN_slowmotion',  # DAIN的网络
                    choices=['DAIN', 'DAIN_slowmotion'], help='model architecture: DAIN | DAIN_slowmotion')
parser.add_argument('-sw', '--save_which', type=int, default=1,  # 保存那个
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


# Detect input type
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
        elif os.path.splitext(files[0])[1].replace('.', '').lower() in ['dpx', 'jpg', 'jpeg', 'exr', 'psd', 'png', 'tif', 'tiff']:
            input_type = 'is'
        else:
            input_type = 'mix'
    return input_type


'''
def fix_name(name):
    target = []
    allow = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
             'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_',
             '-']
    for c in name:
        if c in allow:
            target.append(c)
        else:
            target.append('_')
    return ''.join(target)
'''


def npz2tif(temp_folder):  # 把npz转成tiff
    os.mkdir(f'{temp_folder}/tiff')
    frames_to_process = listdir(f'{temp_folder}/out')
    for frame in frames_to_process:
        img = numpy.load(f'{temp_folder}/out/{frame}')['arr_0']
        cv2.imwrite(f'{temp_folder}/tiff/{frame.split(".")[0]}.tiff', img)


input_type = detect_input_type(args.input)  # 把要处理的一个或多个视频放入一个列表
if input_type == 'mix':
    processes = listdir(args.input)
    processes = [os.path.join(args.input, process) for process in processes]
else:
    processes = [args.input]

for process in processes:  # 为每个视频循环，process是路径
    if input_type == 'continue':  # 如果是从上次的处理中继续处理
        with open(process) as f:
            process_info = eval(f.read())
        batch_size = process_info['batch_size']
        algorithm = process_info['algorithm']
        sf = process_info['sf']
        ffmpeg_dir = process_info['ffmpeg_dir']
        remove_temp_file = process_info['remove_temp_file']
        model_path = process_info['model_path']
        copy_original_frame = process_info['copy_original_frame']
        output = process_info['output']
        output_type = process_info['output_type']
        mac_compatibility = process_info['mac_compatibility']
        project_folder = process_info['project_folder']
        temp_folder = process_info['temp_folder']
        filename = process_info['filename']
        start_frame = len(listdir(f'{temp_folder}/out')) // sf
        if not len(listdir(f'{temp_folder}/out')) % sf:
            start_frame -= 1
        original_frames_to_process = process_info['original_frames_to_process']
        frames_to_process = process_info['original_frames_to_process'][start_frame:]
        copy = process_info['copy']
        net_name = process_info['net_name']
        save_which = process_info['save_which']
    else:
        # Split filename into directory, filename, extension
        filename = os.path.split(process)
        filename = list(filename)
        filename.extend(os.path.splitext(filename[1]))
        filename.pop(1)

        # Create temporary folder
        temp_folder = os.path.join(args.temp_file_path, f'{fix_name(filename[1])}-{random.randint(1000000, 9999999)}')
        temp_folder = os.path.abspath(temp_folder)
        os.makedirs(temp_folder)
        os.makedirs(f'{temp_folder}/out')
        print(f'Created temporary folder: \n{temp_folder}')

        # Detect input type
        input_type = detect_input_type(process)
        # Process input
        if input_type == 'video':
            shutil.copyfile(process, f'{temp_folder}/in{filename[2]}')
            os.makedirs(f'{temp_folder}/in')
            cap = cv2.VideoCapture(f'{temp_folder}/in{filename[2]}')
            frame_count = int(cap.get(7))
            frame_count_len = len(str(frame_count))
            original_fps = cap.get(5)
            for i in range(frame_count):
                rtl, frame = cap.read()
                if rtl:
                    numpy.savez_compressed(f'{temp_folder}/in/{str(i + 1).zfill(frame_count_len)}', frame)
            cap.release()
            # sf
            if args.process_type == 'general':
                sf = args.sf
            elif args.process_type == '60fps':
                sf = int(60 // original_fps)
                if 60 % original_fps:
                    sf += 1
            else:
                sf = 60
            # fps
            if args.fps:
                target_fps = args.fps
            else:
                target_fps = original_fps * sf

        if input_type == 'is':
            shutil.copytree(process, f'{temp_folder}/in_img')
            os.mkdir(f'{temp_folder}/in')
            files = listdir(f'{temp_folder}/in_img')
            for file in files:
                frame = cv2.imread(f'{temp_folder}/in_img/{file}')
                numpy.savez_compressed(f'{temp_folder}/in/{os.path.splitext(file)[0]}', frame)
            frame_count = len(listdir(f'{temp_folder}/in'))
            if not args.fps:
                target_fps = 60
            sf = args.sf

        if input_type == 'npz':
            shutil.copytree(process, f'{temp_folder}/in')
            frame_count = len(listdir(f'{temp_folder}/in'))
            if not args.fps:
                target_fps = 60
            sf = args.sf

        # Check model
        if not os.path.exists(args.model_path):
            print(f"Model {args.model_path} doesn't exist, exiting")
            exit(1)

        # Start/End frame
        copy = True
        start_frame = args.start_frame
        if args.end_frame == 0:
            end_frame = frame_count
        else:
            end_frame = args.end_frame
            copy = False
            # To prevent duplicating the last frame.
            if frame_count == args.end_frame:
                end_frame += 1
                copy = True

        frames_to_process = listdir(f'{temp_folder}/in')
        frames_to_process = frames_to_process[start_frame - 1:end_frame]
        original_frames_to_process = frames_to_process

        # Extract from args
        output = args.output
        output_type = args.output_type
        mac_compatibility = args.mac_compatibility
        ffmpeg_dir = args.ffmpeg_dir
        remove_temp_file = args.remove_temp_file
        start_frame = args.start_frame
        copy_original_frame = args.copy_original_frame
        net_name = args.net_name
        save_which = args.save_which
        algorithm = args.algorithm

    # Log
    project_folder = os.getcwd()
    process_info = {'project_folder': project_folder,
                    'batch_size': args.batch_size,
                    'model_path': args.model_path,
                    'temp_folder': temp_folder,
                    'filename': filename,
                    'sf': sf,
                    'algorithm': algorithm,
                    'original_frames_to_process': original_frames_to_process,
                    'frames_to_process': frames_to_process,
                    'copy_original_frame': copy_original_frame,
                    'output': output,
                    'output_type': output_type,
                    'mac_compatibility': mac_compatibility,
                    'ffmpeg_dir': ffmpeg_dir,
                    'remove_temp_file': remove_temp_file,
                    'copy': copy,
                    'net_name': net_name,
                    'save_which': save_which}
    with open(f'{temp_folder}/process_info.txt', 'w') as f:
        f.write(str(process_info))

    # Process
    os.chdir(temp_folder)
    t = time.time()
    exit_code = os.system(f'{sys.executable} {project_folder}/{run_file_name["algorithm"]}.py')
    print(f'Interpolation spent {round(time.time() - t, 2)}s')
    if exit_code != 0:
        print(exit_code)
        exit(1)
    sf_len = len(str(sf - 1))
    if copy:
        # Copy last frame
        original = f'{temp_folder}/in/{frames_to_process[-1]}'
        for i in range(sf):
            target = f'{temp_folder}/out/{frames_to_process[-1].replace(".npz", "")}_{str(i).zfill(sf_len)}.npz'
            shutil.copyfile(original, target)

    os.chdir(project_folder)

    # Output filename
    if output == 'default':
        # If didn't specify output
        output_dir = f'{filename[0]}/{filename[1]}_{sf}x'
        if output_type == 'video':
            output_dir = f'{output_dir}{filename[2]}'
    else:
        output_dir = output

    # Check output name
    if output_type in ['is', 'npz']:
        if os.path.splitext(output_dir)[1]:
            output_dir = os.path.splitext(output_dir)[0]
    else:  # args.output_type == 'video'
        if not os.path.splitext(output_dir)[1]:
            output_dir = f'{output_dir}.mp4'

    # Check if output directory exists
    if os.path.exists(output_dir):
        output_dir = list(os.path.splitext(output_dir))
        count = 2
        while os.path.exists(f'{output_dir[0]} ({count}){output_dir[1]}'):
            count += 1
        output_dir = f'{output_dir[0]} ({count}){output_dir[1]}'
    print(output_dir)

    # Mac compatibility
    if mac_compatibility:
        pix_fmt = ' -pix_fmt yuv420p'
    else:
        pix_fmt = ''
    if output_type == 'video':
        npz2tif(temp_folder)
        # Check file extension
        if os.path.splitext(output_dir)[1]:
            video_out = f'video{os.path.splitext(output_dir)[1]}'
        elif input_type == 'video':
            video_out = f'video{filename[2]}'
        else:
            video_out = 'video.mp4'
        if input_type == 'video' and start_frame == 1 and start_frame == 0:
            # temp_folder, project_folder, sf, copy, frames_to_process, filename, target_fps, pix_fmt, args.output,output_type,mac_compatibility,ffmpeg_dir,remove_temp_file
            os.system(f'"{os.path.join(ffmpeg_dir, "ffmpeg")}" -loglevel error -thread_queue_size 128 -vsync 0 -r {target_fps} -pattern_type glob -i "{temp_folder}/tiff/*.tiff" -vn -i "{temp_folder}/in{filename[2]}" -vcodec h264{pix_fmt} "{temp_folder}/{video_out}"')
        else:
            print(f'"{os.path.join(ffmpeg_dir, "ffmpeg")}" -loglevel error -vsync 0 -r {target_fps} -pattern_type glob -i "{temp_folder}/tiff/*.tiff" -vcodec h264{pix_fmt} "{temp_folder}/{video_out}"')
            os.system(
                f'"{os.path.join(ffmpeg_dir, "ffmpeg")}" -loglevel error -vsync 0 -r {target_fps} -pattern_type glob -i "{temp_folder}/tiff/*.tiff" -vcodec h264{pix_fmt} "{temp_folder}/{video_out}"')
        shutil.copyfile(f'{temp_folder}/{video_out}', output_dir)
    if output_type == 'is':
        npz2tif(temp_folder)
        shutil.copytree(f'{temp_folder}/tiff', output_dir)
    if output_type == 'npz':
        shutil.copytree(f'{temp_folder}/out', output_dir)

    if remove_temp_file:
        shutil.rmtree(temp_folder)
