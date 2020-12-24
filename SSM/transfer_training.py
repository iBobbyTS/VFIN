# [Super SloMo]
# High Quality Estimation of Multiple Intermediate Frames for Video Interpolation

import os
import shutil
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
from math import log10
import datetime
import time
from tensorboardX import SummaryWriter

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True,
                    help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument("--learning_rate", type=float, default=0.0001,
                    help='set initial learning rate. Default: 0.0001.')

args = parser.parse_args()

# For visualizing loss and interpolated frames

# Function to print and save log
def print_and_log(log):
    print(log)
    log_file = open(os.path.join(os.path.split(args.checkpoint)[0], 'log.txt'), 'a')
    log_file.write(str(log))
    log_file.close()


writer = SummaryWriter('log')

# Create datasets
os.mkdir('extract')
os.mkdir('dataset')
os.system('ffmpeg -loglevel error '
         f"-i '{os.path.join(args.video)}' -vsync 0 "
          "-q:v 2 'extract/%09d.jpg'")
video_frames = os.listdir('extract')
frame_count = len(video_frames)
section_count = frame_count // 12
count = 0
for i in range(section_count):
    os.mkdir(f'dataset/{i}')
    for j in range(12):
        shutil.move(f'extract/{video_frames[count]}', f'dataset/{i}')
        count += 1
shutil.rmtree('extract')

# Initialize flow computation and arbitrary-time flow interpolation CNNs.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = model.UNet(6, 4).to(device)
ArbTimeFlowIntrp = model.UNet(20, 5).to(device)

# Initialze backward warpers for train and validation datasets


trainFlowBackWarp = model.backWarp(352, 352, device).to(device)
validationFlowBackWarp = model.backWarp(640, 352, device).to(device)

# Load Datasets
# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

trainset = dataloader.SuperSloMo(root='dataset', transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
print_and_log(trainset)

# Create transform to display image from tensor
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

# Loss and Optimizer
L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate)
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)

# Initializing VGG16 model for perceptual loss
vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22]).to(device)
for param in vgg16_conv_4_3.parameters():
    param.requires_grad = False


# Initialization
dict1 = torch.load(args.checkpoint)
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])

# Training
start = time.time()
cLoss = dict1['loss']
valLoss = dict1['valLoss']
valPSNR = dict1['valPSNR']
checkpoint_counter = 0

# Main training loop
for epoch in range(dict1['epoch'] + 1, args.epochs):
    print_and_log(f'Epoch: {epoch}\n')
    epoch_start_time = time.time()
    # Append and reset
    cLoss.append([])
    valLoss.append([])
    valPSNR.append([])
    iLoss = 0

    # Increment scheduler count    
    scheduler.step()

    for trainIndex, (trainData, trainFrameIndex) in enumerate(trainloader, 0):

        ## Getting the input and the target from the training set
        frame0, frameT, frame1 = trainData

        I0 = frame0.to(device)
        I1 = frame1.to(device)
        IFrame = frameT.to(device)

        optimizer.zero_grad()

        # Calculate flow between reference frames I0 and I1
        flowOut = flowComp(torch.cat((I0, I1), dim=1))

        # Extracting flows between I0 and I1 - F_0_1 and F_1_0
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]

        fCoeff = model.getFlowCoeff(trainFrameIndex, device)

        # Calculate intermediate flows
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0 = trainFlowBackWarp(I0, F_t_0)
        g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)

        # Calculate optical flow residuals and visibility maps
        intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

        # Extract optical flow residuals and visibility maps
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1 = 1 - V_t_0

        # Get intermediate frames from the intermediate flows
        g_I0_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
        g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)

        wCoeff = model.getWarpCoeff(trainFrameIndex, device)

        # Calculate final intermediate frame 
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                    wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

        # Loss
        recnLoss = L1_lossFn(Ft_p, IFrame)

        prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))

        warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(
            trainFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(trainFlowBackWarp(I1, F_0_1), I0)

        loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(
            torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
        loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(
            torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
        loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

        # Total Loss - Coefficients 204 and 102 are used instead of 0.8 and 0.4
        # since the loss in paper is calculated for input pixels in range 0-255
        # and the input to our network is in range 0-1
        loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth

        # Backpropagate
        loss.backward()
        optimizer.step()
        iLoss += loss.item()

    print_and_log(f'Epoch {epoch} spent {time.time() - epoch_start_time} seconds.\n')
dict1 = {
    'loss': [0], 'valLoss': [0], 'valPSNR': [0], 'epoch': 0,
    'state_dictFC': flowComp.state_dict(),
    'state_dictAT': ArbTimeFlowIntrp.state_dict(),
}
model_save_path = f'{args.checkpoint}_new.pth'
torch.save(dict1, model_save_path)
print_and_log(f'Saved epoch {epoch} to {model_save_path}.\n\n')
