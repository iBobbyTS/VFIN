import warnings

import numpy
import torch
from torch.autograd import Variable

import networks as networks
from empty_cache import empty_cache


class Interpolator:
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    def __init__(self, model_directory: str, sf: int, height: int, width: int, **dain):
        # args
        self.save_which = dain['save_which']
        # Model
        model = networks.__dict__[dain['net_name']](
            channel=3, filter_size=4, timestep=1 / sf, training=False).cuda()
        empty_cache()

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(model_directory).items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        del pretrained_dict, model_dict
        self.model = model.eval()

        # ndarray2tensor
        if dain['channel'] == 3:
            self.ndarray2tensor = lambda frame: torch.cuda.ByteTensor(frame).permute(2, 0, 1).float() / 255
        elif dain['channel'] == 4:
            self.ndarray2tensor = lambda frame: torch.cuda.ByteTensor(frame)[:, :, :3].permute(2, 0, 1).float() / 255

        # pader
        if width != ((width >> 7) << 7):
            intWidth_pad = (((width >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft = int((intWidth_pad - width) / 2)
            intPaddingRight = intWidth_pad - width - intPaddingLeft
        else:
            intPaddingLeft = 32
            intPaddingRight = 32

        if height != ((height >> 7) << 7):
            intHeight_pad = (((height >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - height) / 2)
            intPaddingBottom = intHeight_pad - height - intPaddingTop
        else:
            intPaddingTop = 32
            intPaddingBottom = 32

        self.pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])
        self.hs = intPaddingLeft  # Horizontal start
        self.he = intPaddingLeft + width
        self.vs = intPaddingTop
        self.ve = intPaddingTop + height  # Vertical end

    def interpolate(self, frames):
        X0 = self.pader(Variable(torch.unsqueeze(self.ndarray2tensor(frames[0]), 0)))
        X1 = self.pader(Variable(torch.unsqueeze(self.ndarray2tensor(frames[1]), 0)))
        empty_cache()

        y_ = self.model(torch.stack((X0, X1), dim=0))[0]
        empty_cache()
        y_ = y_[self.save_which]
        y_ = [[(255*item).clamp(0.0, 255.0).byte()[0, :, self.vs:self.ve,self.hs:self.he]
                        .permute(1, 2, 0).cpu().numpy()] for item in y_]
        empty_cache()

        return y_
