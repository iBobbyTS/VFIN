import argparse
import os
from shutil import rmtree, move
import random
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--ffmpeg_dir', type=str, default='', help='path to ffmpeg.exe')
parser.add_argument('--videos_folder', type=str, required=True, help='path to the folder containing videos')
parser.add_argument('--dataset_folder', type=str, required=True, help='path to the output dataset folder')
parser.add_argument('--continue_from', type=str, default='None', choices=['None', 'Folder', 'tar'],
                    help='path to the output dataset folder')
parser.add_argument('--dest_tar', type=str, help='path to the output tar')
parser.add_argument('--img_width', type=int, default=360, help='output image width')
parser.add_argument('--img_height', type=int, default=640, help='output image height')
parser.add_argument('--train_test_val_split', type=tuple, default=(70, 20, 10),
                    help='train test split for custom dataset')

args = parser.parse_args()

if args.videos_folder[-1] == '/':
    args.videos_folder = args.videos_folder[:-1]
if args.dataset_folder[-1] == '/':
    args.dataset_folder = args.dataset_folder[:-1]
if args.dest_tar and args.dest_tar[-1] == '/':
    args.dest_tar = args.dest_tar[:-1]


def listdir(path):
    files = os.listdir(path)
    for file in ['.DS_Store']:
        if file in files:
            files.remove(file)
    files.sort()
    return files


extractPath = os.path.join(args.dataset_folder, 'extracted')
trainPath = os.path.join(args.dataset_folder, 'train')
testPath = os.path.join(args.dataset_folder, 'test')
validationPath = os.path.join(args.dataset_folder, 'validation')


if args.continue_process != 'Folder' and os.path.exists(args.dataset_folder):
    if input('dataset_folder exists, delete? [y, n]: ').lower() == 'y':
        rmtree(args.dataset_folder)
    else:
        exit(1)
os.makedirs(args.dataset_folder)
if args.continue_process != 'Folder':
    for folder in ['train', 'test', 'validation']:
        os.mkdir(f'{args.dataset_folder}/{folder}')
os.mkdir(f'{args.dataset_folder}/extracted')

videos = listdir(args.videos_folder)
video_frames = {}
for i, video in enumerate(videos):
    video_extraction_path = os.path.join(extractPath, video.split('.')[0])
    os.mkdir(video_extraction_path)
    os.system(f"{os.path.join(args.ffmpeg_dir, 'ffmpeg')} -loglevel error "
              f"-i '{os.path.join(args.videos_folder, video)}' -vsync 0 -s 50x50 "
              f"-q:v 2 '{video_extraction_path}/%09d.jpg'")
    video_frames[video] = listdir(video_extraction_path)
    print(f'\rProcessed {i+1}/{len(videos)}: {video}', end='', flush=True)
print()
if args.continue_process == 'Folder':
    ori_val_count = len(listdir(f'{args.dataset_folder}/validation'))
    ori_test_count = len(listdir(f'{args.dataset_folder}/test'))
    ori_train_count = len(listdir(f'{args.dataset_folder}/train'))
    counts = [ori_val_count, ori_test_count, ori_train_count]
    ori_total_section_count = ori_val_count + ori_test_count + ori_train_count

elif args.continue_process == 'tar':
    counts = []
    for tar in ['validation', 'test', 'train']:
        current = []
        with tarfile.open(f'{args.dest_tar}/{tar}.tar') as f:
            files = f.getnames()
        for file in files:
            if '/' not in file:
                current.append(file)
        counts.append(len(current))
    ori_val_count, ori_test_count, ori_train_count = counts
    ori_total_section_count = ori_val_count + ori_test_count + ori_train_count
else:
    ori_total_section_count = 0
    ori_val_count = 0
    ori_test_count = 0
    ori_train_count = 0
total_section_count = ori_total_section_count + sum([len(i) // 12 for i in video_frames.values()])
val_count = int(total_section_count * (args.train_test_val_split[2] / 100)) - ori_val_count
test_count = int(total_section_count * (args.train_test_val_split[1] / 100)) - ori_test_count
train_count = total_section_count - ori_val_count - ori_test_count - ori_train_count
total_section_count -= ori_total_section_count
total_section = range(total_section_count)
train_set = list(total_section)
val_set = random.sample(train_set, val_count)
for tmp in val_set:
    train_set.remove(tmp)
test_set = random.sample(train_set, test_count)
for tmp in test_set:
    train_set.remove(tmp)

video_frames = list(video_frames.values())
if args.continue_process != 'None':
    val_test_train_count = counts
else:
    val_test_train_count = [0, 0, 0]

for section_index, section in enumerate(total_section):
    if len(video_frames[0]) < 12:
        video_frames.pop(0)
        videos.pop(0)
    frames = video_frames[0][:12]
    if random.randint(0, 1):
        frames = frames[::-1]

    # dest = 'None'
    if section in val_set:
        dest = f'validation/{val_test_train_count[0]}'
        val_test_train_count[0] += 1
    if section in test_set:
        dest = f'test/{val_test_train_count[1]}'
        val_test_train_count[1] += 1
    if section in train_set:
        dest = f'train/{val_test_train_count[2]}'
        val_test_train_count[2] += 1

    os.mkdir(f'{args.dataset_folder}/{dest}')
    for i, frame in enumerate(frames):
        move(f"{args.dataset_folder}/extracted/{videos[0].split('.')[0]}/{frame}",
             f'{args.dataset_folder}/{dest}/{i}.jpg')
        video_frames[0].remove(frame)
    print(f'\r{section_index + 1}/{total_section_count}', end='', flush=True)
print()
rmtree(extractPath)

# Tar
if args.dest_tar:
    if args.continue_process != 'tar':
        os.makedirs(args.dest_tar)
    for folder in ['validation', 'test', 'train']:
        tar = tarfile.open(f'{args.dest_tar}/{folder}.tar', 'a')
        for sub in listdir(f'{args.dataset_folder}/{folder}'):
            tar.add(f'{args.dataset_folder}/{folder}/{sub}', sub)
        tar.close()
    rmtree(args.dataset_folder)

