# VFIN (Video Frame INterpolation)
Combination of [DAIN](https://github.com/baowenbo/DAIN), [Super SloMo(SSM)](https://github.com/avinashpaliwal/Super-SloMo), [BIN](https://github.com/laomao0/BIN) and more coming together. Now SSM is developed, DAIN is almost done and BIN is still be working on. 

### Table of Contents
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Inferencing](#easy-inferencing)


### Citation
[Frame interpolation for normal video](https://github.com/baowenbo/DAIN/)
    @inproceedings{DAIN,
        author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
        title     = {Depth-Aware Video Frame Interpolation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2019}
    }

[Frame interpolation for general video](https://github.com/avinashpaliwal/Super-SloMo)
    @inproceedings{Super SloMo,
        author    = {Jiang, Huaizu and Sun, Deqing and Jampani, Varun and Yang, Ming-Hsuan and Learned-Miller, Erik and Kautz, Jan},
        title     = {High Quality Estimation of Multiple Intermediate Frames for Video Interpolation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2018}
    }

[Frame interpolation for blurry video](https://github.com/laomao0/BIN)
     @inproceedings{BIN,
        author    = {Shen, Wang and Bao, Wenbo and Zhai, Guangtao and Chen, Li and Min, Xiongkuo and Gao, Zhiyong}, 
        title     = {Blurry Video Frame Interpolation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2020}
    }


### Requirements and Dependencies
- General
	- FFmpeg
	- NVIDIA GPU (We test with: P100, P4, T4, K80)
- DAIN
	- Ubuntu (We test with Ubuntu = 18.04.5 LTS)
	- Python (We test with: Python = 3.6.12)
	- Cuda and/or Cudnn (We test with: - Cuda = 10.1)
	- PyTorch >= 1.0.0, <= 1.4.0
	- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 9.0 compilers)
- SSM
	- Ubuntu (We test with Ubuntu = 18.04.5 LTS)
	- Python (We test with Python = 3.6.12)
	- PyTorch >= 0.4.1 (We test with: PyTorch = 1.6.0)


### Installation
Install dependencies:

	# Python
	pip install Pillow numpy opencv-python click tensorboardX future h5py dominate scikit-image==0.16.2 scipy==1.0.0 functions visdom
	# Change the cuda version to your version. ex. CUDA 9.2: +cu92, or CPU, +cpu
	pip install torch==1.4.0+cu101 torchvision==0.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
	
	# Conda (Anaconda/Miniconda)
	conda install Pillow numpy click future h5py scipy==1.0.0 scikit-image==0.16.12 -y
	pip install dominate opencv-python functions visdom opencv-python tensorboardX
	# Change cudatoolkit to your version
	conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch -y
	
Download repository:

	git clone https://github.com/iBobbyTS/VFIN.git
	cd VFIN
    
Generate our PyTorch extensions: (This will take approximately 20 minuets)
    
	cd DAIN
	./build.sh
	cd ..
    
Download pretrained models, 

	# DAIN
	mkdir DAIN/model_weights
	wget http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth -O DAIN/model_weights/best.pth
	# SSM (wget might not work, find other wayto download it from Google Drive and copy it to SSM)
	wget https://drive.google.com/file/d/1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF/view?usp=drive_open -O SSM/SuperSloMo.ckpt

### Easy inferencing

	python run.py -i input.mp4

Check Other arguements in [run.py](https://github.com/iBobbyTS/VFIN/blob/master/run.py). 

### Contact
[iBobby](mailto:iBobbyTS@gmail.com)

### License
See [MIT License](https://github.com/iBobby/VFIN/blob/master/LICENSE)
