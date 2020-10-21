import os
import shutil
import cv2
import numpy


def listdir(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    files.sort()
    return files


def calculate_splition(height, width, y_split=1, x_split=1):
    parts = {}
    for x_count in range(x_split):
        x_start = int((x_count - 0.1) * width / x_split) if x_count != 0 else 0
        x_end = int((x_count + 1.1) * width / x_split) if x_count + 1 != x_split else width
        for y_count in range(y_split):
            y_start = int((y_count - 0.1) * height / y_split) if y_count != 0 else 0
            y_end = int((y_count + 1.1) * height / y_split) if y_count + 1 != y_split else height
            parts[f'{x_count}_{y_count}'] = ((y_start, y_end, x_start, x_end))
    return parts


def split_is(in_path, out_path, x_split, y_split):
    shutil.rmtree(out_path)
    os.makedirs(out_path)
    for i in range(x_split):
        for j in range(y_split):
            os.makedirs(f'{out_path}/{i}_{j}')
    files = listdir(in_path)
    height, width = cv2.imread(f'{in_path}/{files[0]}').shape[0:2]
    splitions = calculate_splition(height, width, 4, 2)
    count_length = len(str(len(files)))
    for file_index, file in enumerate(files):
        img = cv2.imread(f'{in_path}/{file}')
        out_imgs = [img[ys:ye, xs:xe] for ys, ye, xs, xe in splitions.values()]
        for coordinate, out_img in zip(splitions.keys(), out_imgs):
            cv2.imwrite(f'{out_path}/{coordinate}/{str(file_index).zfill(count_length)}.tiff', out_img)
        print(f'\r{file_index + 1}/{len(files)}', end='', flush=True)


# split_is('/Users/ibobby/Dataset/resolution_test/4k',
#          '/Users/ibobby/Dataset/resolution_test/4k_out',
#          2, 4)


def combine_is(in_path, out_path, x_split, y_split, height, width):
    os.makedirs(out_path)
    part_x, part_y = int(width / x_split), int(height / y_split)

    folder_names = []
    for y_count in range(y_split):
        for x_count in range(x_split):
            folder_names.append(f'{x_count}_{y_count}')

    locating_t = []
    splition = calculate_splition(width, height, x_split, y_split)
    for original_size in splition.values():
        original_size = list(original_size)
        original_size[0] = int(original_size[0] + part_x / 10) if original_size[0] != 0 else 0
        original_size[1] = int(original_size[1] - part_x / 10) if original_size[1] != width else width
        original_size[2] = int(original_size[2] + part_y / 10) if original_size[2] != 0 else 0
        original_size[3] = int(original_size[3] - part_y / 10) if original_size[3] != height else height
        locating_t.append(original_size)

    locating_o = []
    for part, locating_t_ in zip(folder_names, locating_t):
        xs = int(part_x / 10) if locating_t_[0] != 0 else 0
        xe = int(part_x * 1.1) if locating_t_[1] != part_x else part_x
        ys = int(part_y / 10) if locating_t_[2] != 0 else 0
        ye = int(part_y * 1.1) if locating_t_[3] != part_y else part_y
        locating_o.append((xs, xe, ys, ye))

    files = sorted(listdir(f'{in_path}/0_0'))
    file_count = len(files)
    for frame_name in files:
        out_img = numpy.zeros((height, width, 3), 'uint8')
        for folder_name, lt, lo in zip(folder_names, locating_t, locating_o):
            replace_img = cv2.imread(f'{in_path}/{folder_name}/{frame_name}')
            out_img[lt[2]:lt[3], lt[0]:lt[1]] = replace_img[lo[2]:lo[3], lo[0]:lo[1]]
        cv2.imwrite(f'{out_path}/{frame_name}', out_img)
        print(f'\r{int(frame_name.split(".")[0]) + 1}/{file_count}', end='', flush=True)

