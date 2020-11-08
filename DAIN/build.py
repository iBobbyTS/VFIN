import os
import sys
import argparse
import torch
import json


parser = argparse.ArgumentParser()
parser.add_argument('-bt', '--build_type', type=str, choices=['install', 'develop', 'bdist_wheel'], default='install')
parser.add_argument('-cc', '--compute_compatibility', type=str, help='ex: 60,61,75')
args = parser.parse_args()

python_executable = sys.executable
print(f'Building CUDAExtension for PyTorch in {python_executable}')
torch_version = torch.__version__
torch_version_split = torch_version.split('.')
prefix = 'You need torch>=1.0.0, <=1.4.0, you have torch=='
if torch_version_split[0] is '0':
    raise RuntimeError(prefix + torch_version + ' < 1.0.0')
elif int(torch_version_split[0]) > 1 or int(torch_version_split[1]) > 4:
    raise RuntimeError(prefix + torch_version + ' > 1.4.0')

nvcc_args = []
for cc in args.compute_compatibility.split(','):
    nvcc_args.append('-gencode')
    nvcc_args.append(f'arch=compute_{cc},code=sm_{cc}')
nvcc_args.append('-w')
with open('compiler_args.json', 'w') as f:
    json.dump({'nvcc': nvcc_args, 'cxx': ['-std=c++11', '-w']}, f)
print(f'Compiling for compute compatilility {args.compute_compatibility}')

os.chdir('my_package')

folders = [folder for folder in sorted(os.listdir('.')) if os.path.isdir(folder)]

for folder in folders[0:1]:
    os.chdir(f"{'' if folder == folders[0] else '../'}{folder}")
    os.system(f'{python_executable} setup.py {args.build_type}')

# os.chdir('../../PWCNet/correlation_package_pytorch1_0')
# os.system(f'{python_executable} setup.py {args.build_type}')

