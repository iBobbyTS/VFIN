import warnings
import math

import numpy
import torch

import model

warnings.filterwarnings("ignore")


class Interpolator:
    def __init__(self, model_directory: str, sf: int, height: int, width: int, batch_size=1, **ssm):
        # sf
        self.sf = sf
        self.batch_size = batch_size

        # Check if need to expand image
        self.h_w = [int(math.ceil(height / 32) * 32 - height) if height % 32 else 0,
                    int(math.ceil(width / 32) * 32) - width if width % 32 else 0]
        dim = [height + self.h_w[0], width + self.h_w[1]]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.flowComp = model.UNet(6, 4)

        self.flowComp.to(device)

        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = model.UNet(20, 5)
        self.ArbTimeFlowIntrp.to(device)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False
        self.flowBackWarp = model.backWarp(dim[1], dim[0], device)
        self.flowBackWarp = self.flowBackWarp.to(device)
        dict1 = torch.load(model_directory, map_location='cpu')
        self.ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
        self.flowComp.load_state_dict(dict1['state_dictFC'])
        cuda_availability = torch.cuda.is_available()
        self.ndarray2tensor = {True: self.ndarray2cuda_tensor,
                               False: self.ndarray2cpu_tensor
                               }[cuda_availability]
        self.tensor2ndarray = {True: None,
                               False: lambda frames: numpy.transpose((numpy.array(frames) / 255).astype(numpy.uint8)
                                                                     [:, ::-1, self.h_w[0]:, self.h_w[1]:], (0, 2, 3, 1))
                               }[cuda_availability]

    def ndarray2cuda_tensor(self, frames: list):  # 内部调用
        out_frames = []
        for frame in frames:
            frame = torch.cuda.ByteTensor(frame[:, :, ::-1].copy())
            frame = torch.cat([torch.zeros((frame.shape[0], self.h_w[1], 3), dtype=torch.cuda.uint8), frame], dim=1)
            frame = torch.cat([torch.zeros((self.h_w[0], frame.shape[1], 3), dtype=torch.cuda.uint8), frame], dim=0)
            out_frames.append(frame.permute(2, 0, 1))
        return torch.stack(out_frames, dim=0).float() / 255

    def ndarray2cpu_tensor(self, frames: list):
        out_frames = []
        for frame in frames:
            frame = numpy.insert(frame, 0, numpy.zeros((self.h_w[1], frame.shape[0], 3), numpy.uint8), 1)
            frame = numpy.insert(frame, 0, numpy.zeros((self.h_w[0], frame.shape[1], 3), numpy.uint8), 0)
            out_frames.append(frame)
        return torch.Tensor(numpy.transpose(numpy.array(out_frames), (0, 3, 1, 2))[:, ::-1].astype('float32') / 255)

    def interpolate(self, frames: list):
        with torch.no_grad():
            I0 = self.ndarray2tensor(frames[:-1])
            I1 = self.ndarray2tensor(frames[1:])
            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            intermediate_frames = []  # Each item contains intermediate frames
            for intermediateIndex in range(1, self.sf):
                t = float(intermediateIndex) / self.sf
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                        wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                # Ft_p contains batches of one intermediate frame
                intermediate_frames.append(self.tensor2ndarray(Ft_p))
        return intermediate_frames
