import os
import shutil
import time
import math
import warnings
import torch
import numpy
import SSM.model as model

warnings.filterwarnings("ignore")


def main(process_info_path):
    def file_to_tensor(frame_dir, all_frames_to_process, frame_range, w_h):
        files = all_frames_to_process[frame_range[0]: frame_range[1]]
        frames = []
        for file in files:
            frame = numpy.load(f'{frame_dir}/{file}')['arr_0']
            frame = numpy.transpose(frame, (2, 0, 1))
            frame = frame[::-1, :, :]
            for s in w_h:
                if w_h[s]:
                    if s == 'w':
                        frame = numpy.insert(frame, 0, numpy.zeros((w_h[s], 3, frame.shape[1]), 'uint8'), 2)
                    else:  # s == 'h'
                        frame = numpy.insert(frame, 0, numpy.zeros((w_h[s], 3, frame.shape[2]), 'uint8'), 1)
            frames.append(frame)
        frames = numpy.array(frames).astype('float32')
        frames = torch.from_numpy(frames)  # .cuda()
        frames = frames / 255
        return frames

    with open(process_info_path, 'r') as file:
        process_info = file.read()
        process_info = eval(process_info)

    # Length
    sf_length = len(str(process_info['sf'] - 1))

    # Check if need to expand image
    ori_dim = numpy.load(f'{process_info["current_temp_file_path"]}/in/{process_info["frames_to_process"][0]}')[
        'arr_0'].shape
    h_w = {'h': math.ceil(ori_dim[0] / 32) * 32 - ori_dim[0] if ori_dim[0] % 32 else 0,
           'w': math.ceil(ori_dim[1] / 32) * 32 - ori_dim[1] if ori_dim[1] % 32 else 0}
    dim = [ori_dim[0] + h_w['h'], ori_dim[1] + h_w['w']]
    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    flowComp = model.UNet(6, 4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False
    ArbTimeFlowIntrp = model.UNet(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backWarp(dim[1], dim[0], device)
    flowBackWarp = flowBackWarp.to(device)
    dict1 = torch.load(process_info['model_path'], map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
    # Interpolate frames
    loop_timer = []
    try:
        with torch.no_grad():
            batch_count = len(process_info['frames_to_process']) // process_info['batch_size'] - 1
            # frame1 = file_to_tensor(f'{process_info["current_temp_file_path"]}/in',
            #                     [0, process_info['batch_size'] if process_info['batch_size'] < len(
            #                         process_info['frames_to_process']) else len(process_info['frames_to_process'])],
            #                     h_w)
            I1 = file_to_tensor(f'{process_info["current_temp_file_path"]}/in', process_info['frames_to_process'],
                                [0, process_info['batch_size'] if process_info['batch_size'] < len(
                                    process_info['frames_to_process']) else len(process_info['frames_to_process'])],
                                h_w).to(device)

            for _ in range(batch_count):
                batch_start_time = time.time()
                frame_range = [_ * process_info['batch_size'], (_ + 1) * process_info['batch_size']]
                if frame_range[1] > len(process_info['frames_to_process']):
                    frame_range[1] = len(process_info['frames_to_process'])
                I0 = I1
                I1 = file_to_tensor(f'{process_info["current_temp_file_path"]}/in', process_info['frames_to_process'],
                                    [i + 1 for i in frame_range], h_w).to(device)
                x = torch.cat((I0, I1), dim=1)
                flowOut = flowComp(x)
                F_0_1 = flowOut[:, :2, :, :]
                F_1_0 = flowOut[:, 2:, :, :]
                # Save reference frames in output folder
                for batchIndex in range(len(I0)):
                    shutil.copy(
                        f'{process_info["current_temp_file_path"]}/in/{process_info["frames_to_process"][_ * process_info["batch_size"] + batchIndex]}',
                        f'{process_info["current_temp_file_path"]}/out/{process_info["frames_to_process"][_ * process_info["batch_size"] + batchIndex].replace(".npz", "")}_{"0".zfill(sf_length)}.npz')

                # Generate intermediate frames
                for intermediateIndex in range(1, process_info['sf']):
                    t = float(intermediateIndex) / process_info['sf']
                    temp = -t * (1 - t)
                    fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                    F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                    F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                    g_I0_F_t_0 = flowBackWarp(I0, F_t_0)
                    g_I1_F_t_1 = flowBackWarp(I1, F_t_1)

                    intrpOut = ArbTimeFlowIntrp(
                        torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                    F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                    F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                    V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                    V_t_1 = 1 - V_t_0

                    g_I0_F_t_0_f = flowBackWarp(I0, F_t_0_f)
                    g_I1_F_t_1_f = flowBackWarp(I1, F_t_1_f)

                    wCoeff = [1 - t, t]

                    Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                            wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                    # Save intermediate frame
                    for batchIndex in range(len(Ft_p)):
                        img = Ft_p[batchIndex].cpu().detach()
                        img = numpy.array(img)
                        img = img * 255
                        img = numpy.round(img)
                        img = numpy.transpose(img, (1, 2, 0))
                        img = img[:, :, ::-1]  # RGB -> BGR
                        img = img[h_w['h']:, h_w['w']:, :]
                        img = img.astype('uint8')
                        numpy.savez_compressed(
                            f'{process_info["current_temp_file_path"]}/out/{process_info["frames_to_process"][_ * process_info["batch_size"] + batchIndex].replace(".npz", "")}_{str(intermediateIndex).zfill(sf_length)}',
                            img)

                time_spent = time.time() - batch_start_time
                if _ == 0:
                    batch_count_len = len(str(batch_count))
                    batch_count_str = str(batch_count).zfill(batch_count_len)
                    len_time_spent = len(str(round(time_spent))) + 5
                    if process_info['batch_size'] > 1:
                        batch_or_frame = 'frame'
                    else:
                        batch_or_frame = 'batch'
                loop_timer.append(time_spent)
                batches_left = batch_count - _

                estimated_seconds_left = round(batches_left * sum(loop_timer) / len(loop_timer), 2)
                m, s = divmod(estimated_seconds_left, 60)
                h, m = divmod(m, 60)
                estimated_time_left = "%d:%02d:%02d" % (h, m, s)
                print('\r' +
                      f"*** Processed {batch_or_frame} {str(_ + 1).zfill(batch_count_len)}/{batch_count_str} | Time spent: {(str(round(time_spent, 2)) + 's').ljust(len_time_spent)} | Time left: {estimated_time_left} *****",
                      end='', flush=True)
        print('\nFinished processing images.')
    except KeyboardInterrupt:
        exit(1)
