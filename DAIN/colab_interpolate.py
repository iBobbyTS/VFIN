import time
import os
from torch.autograd import Variable
import torch
import numpy
import networks as networks
import shutil
import warnings


def run(process_info_path):
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.backends.cudnn.benchmark = True

    with open(process_info_path, 'r') as file:
        process_info = file.read()
        process_info = eval(process_info)
    os.chdir(process_info['project_folder'])
    sf_length = len(str(process_info['sf'] - 1))

    model = networks.__dict__[process_info['net_name']](
        channel=3,
        filter_size=4,
        timestep=1 / process_info['sf'],
        training=False).cuda()

    model_path = process_info['model_path']

    pretrained_dict = torch.load(model_path)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []

    model = model.eval()  # deploy mode

    timestep = 1 / process_info['sf']
    time_offsets = [kk * timestep for kk in range(1, int(1.0 / timestep))]

    torch.set_grad_enabled(False)

    input_files = process_info['frames_to_process']
    loop_timer = []
    try:
        for _ in range(len(input_files) - 1):

            start_time = time.time()

            filename_frame_1 = f'{process_info["temp_folder"]}/in/{input_files[_]}'
            filename_frame_2 = f'{process_info["temp_folder"]}/in/{input_files[_ + 1]}'

            # X0 = torch.from_numpy(numpy.transpose(numpy.load(filename_frame_1)['arr_0'], (2, 0, 1))[0:3].astype("float32") / 255.0).type(torch.cuda.FloatTensor)
            # X1 = torch.from_numpy(numpy.transpose(numpy.load(filename_frame_2)['arr_0'], (2, 0, 1))[0:3].astype("float32") / 255.0).type(torch.cuda.FloatTensor)
            if _ == 0:
                X0 = torch.cuda.FloatTensor(numpy.load(filename_frame_1)['arr_0'])[:, :, :3].permute(2, 0, 1) / 255
            else:
                X0 = X1_ori
            X1 = torch.cuda.FloatTensor(numpy.load(filename_frame_2)['arr_0'])[:, :, :3].permute(2, 0, 1) / 255
            X1_ori = X1

            assert (X0.size() == X1.size())

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channels = X0.size(0)
            if not channels == 3:
                print(
                    f"Skipping {filename_frame_1}-{filename_frame_2} -- expected 3 color channels but found {channels}.")
                continue

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft = int((intWidth_pad - intWidth) / 2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intPaddingLeft = 32
                intPaddingRight = 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            X0 = Variable(torch.unsqueeze(X0, 0))
            X1 = Variable(torch.unsqueeze(X1, 0))
            X0 = pader(X0)
            X1 = pader(X1)

            y_s, offset, filter = model(torch.stack((X0, X1), dim=0))
            y_ = y_s[process_info['save_which']]

            X0 = X0.data.cpu().numpy()
            if not isinstance(y_, list):
                y_ = [y_.data.cpu().numpy()]
            else:
                y_ = [item.data.cpu().numpy() for item in y_]
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()
            X0 = numpy.transpose(255.0 * X0.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                         intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))
            y_ = [numpy.transpose(255.0 * item.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                          intPaddingLeft:intPaddingLeft + intWidth], (1, 2, 0)) for item in y_]
            offset = [numpy.transpose(
                offset_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for offset_i in offset]
            filter = [numpy.transpose(
                filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                (1, 2, 0)) for filter_i in filter] if filter is not None else None
            X1 = numpy.transpose(255.0 * X1.clip(0, 1.0)[0, :, intPaddingTop:intPaddingTop + intHeight,
                                         intPaddingLeft: intPaddingLeft + intWidth], (1, 2, 0))

            interpolated_frame_number = 0
            shutil.copy(filename_frame_1,
                        f'{process_info["temp_folder"]}/out/{input_files[_].replace(".npz", "")}_{"0".zfill(sf_length)}.npz')
            for item, time_offset in zip(y_, time_offsets):
                interpolated_frame_number += 1
                output_frame_file_path = f'{process_info["temp_folder"]}/out/{input_files[_].replace(".npz", "")}_{str(interpolated_frame_number).zfill(sf_length)}'
                numpy.savez_compressed(output_frame_file_path, numpy.round(item).astype('uint8'))

            end_time = time.time()
            time_spent = end_time - start_time
            if _ == 0:
                frame_count_len = len(str(len(input_files)))
                print(
                    f"****** Initialized model and processed frame {'1'.zfill(frame_count_len)} | Time spent: {round(time_spent, 2)}s ******************")
            else:
                if _ == 1:
                    len_time_spent = len(str(round(time_spent))) + 5
                loop_timer.append(time_spent)
                frames_left = len(input_files) - _ - 2
                estimated_seconds_left = round(frames_left * sum(loop_timer) / len(loop_timer), 2)
                m, s = divmod(estimated_seconds_left, 60)
                h, m = divmod(m, 60)
                estimated_time_left = "%d:%02d:%02d" % (h, m, s)
                print(
                    f"****** Processed frame {str(_ + 1).zfill(frame_count_len)} | Time spent: {(str(round(time_spent, 2)) + 's').ljust(len_time_spent)} | Time left: {estimated_time_left} ******************")

        print("Finished processing images.")
    except KeyboardInterrupt:
        exit(1)
