# VFIN (Video Frame INterpolation)
Combination of [DAIN](https://github.com/baowenbo/DAIN), [Super SloMo(SSM)](https://github.com/avinashpaliwal/Super-SloMo), [BIN](https://github.com/laomao0/BIN) and more coming together. Now SSM is developed, DAIN is almost done and BIN is still be working on. 

### Table of Contents
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)


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
	- NVIDIA GPU (We test with: [These GPUs](https://github.com/iBobbyTS/VFIN/GPU)
- DAIN
	- Ubuntu (We test with:
		- Ubuntu = 16.04.5 LTS
		- Ubuntu = 18.04.5 LTS)
	- Python (We test with: 
		- Python = 3.6.8 in Anaconda3 = 4.1.1
		- Python = 3.6.12)
	- Cuda and/or Cudnn (We test with:
		- Cuda = 9.0 and Cudnn = 7.0
		- Cuda = 10.1)
	- PyTorch (The customized depth-aware flow projection and other layers require ATen API in PyTorch = 1.0.0) (We test with:
		- PyTorch = 1.0.0
		- Pytorch = 1.4.0)
	- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 9.0 compilers)
	- NVIDIA GPU (We use Titan X (Pascal) with compute = 6.1, but we support compute_50/52/60/61 devices, should you have devices with higher compute capability, please revise [this](https://github.com/baowenbo/DAIN/blob/master/my_package/DepthFlowProjection/setup.py))
- SSM
	- Ubuntu (We test with Ubuntu = 18.04.5 LTS)
	- Python (We test with Python = 3.6.12)
	- PyTorch >= 0.4.1 (We test with:
		- Pytorch = 1.4.0
		- PyTorch = 1.6.0)
	- If you're using GPU
		- Cuda (We test with:
			- Cuda = 9.2
			- Cuda = 10.1)
		- NVIDIA GPU
