#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F

### custom lib
import utils
import matplotlib.pyplot as plt
from networks.rtrwnet import Model, flow_warping

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')

    parser.add_argument('-model_name', type=str, required=True, help='test model name')
    parser.add_argument('-epoch', type=int, required=True, help='epoch')
    parser.add_argument('-streak_tag',          type=str,     default=1,               help='Whether the model handle rain streak')
    parser.add_argument('-haze_tag',            type=str,     default=1,               help='Whether the model handle haze')
    parser.add_argument('-flow_tag',            type=str,     default=1,               help='Whether the model handle haze')

    parser.add_argument('-data_dir', type=str, default='data', help='path to data folder')
    parser.add_argument('-output_dir', type=str, default='result', help='path to list folder')
    parser.add_argument('-file_suffix', type=str, default='.jpg', help='path to list folder')


    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint folder')
    parser.add_argument('-list_filename', type=str, required=True, help='evaluated task')
    parser.add_argument('-redo', action="store_true", help='Re-generate results')
    parser.add_argument('-gpu', type=int, default=1, help='gpu device id')

    parser.add_argument('-ifAlign', type=int, default=1, help='')
    parser.add_argument('-ifAggregate', type=int, default=1, help='')

    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 3  ## Inputs to TransformNet need to be divided by 4

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")

    model_filename = os.path.join(opts.checkpoint_dir, opts.model_name, "model_epoch_%d.pth" % opts.epoch)
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)

    opts.rgb_max = 1.0
    opts.fp16 = False

    Net = Model(opts)
    def remove_state_module(state_dict):
        state_dict_wo_module = {}
        for key in state_dict:
            key_wo_module = key.replace('.module.', '.')
            state_dict_wo_module[key_wo_module] = state_dict[key]
        return state_dict_wo_module
    Net.load_state_dict(remove_state_module(state_dict['three_dim_model']))

    device = torch.device("cuda" if opts.cuda else "cpu")
    Net = Net.cuda()

    Net.eval()


    list_filename = opts.list_filename

    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    times = []

    for v in range(len(video_list)//2):
        video = video_list[v*2]

        print("Test on video %d/%d: %s" % (v + 1, len(video_list), video))

        input_dir = os.path.join(opts.data_dir,  video)
        output_dir = os.path.join(opts.output_dir, opts.model_name, "epoch_%d" % opts.epoch, video)

        print(input_dir)
        print(output_dir)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        frame_list = sorted(glob.glob(os.path.join(input_dir, "*"+opts.file_suffix)))
        output_list = glob.glob(os.path.join(output_dir, "*"+opts.file_suffix))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue

        for t in range(3, len(frame_list)-3):
            frame_i0 = utils.read_img(frame_list[t-3])
            frame_i1 = utils.read_img(frame_list[t-2])
            frame_i2 = utils.read_img(frame_list[t-1])
            frame_i3 = utils.read_img(frame_list[t])
            frame_i4 = utils.read_img(frame_list[t+1])
            frame_i5 = utils.read_img(frame_list[t+2])
            frame_i6 = utils.read_img(frame_list[t+3])

            with torch.no_grad():
                frame_i0 = utils.img2tensor(frame_i0).cuda()
                frame_i1 = utils.img2tensor(frame_i1).cuda()
                frame_i2 = utils.img2tensor(frame_i2).cuda()
                frame_i3 = utils.img2tensor(frame_i3).cuda()
                frame_i4 = utils.img2tensor(frame_i4).cuda()
                frame_i5 = utils.img2tensor(frame_i5).cuda()
                frame_i6 = utils.img2tensor(frame_i6).cuda()

                frame_i0, f_h_pad, f_w_pad = utils.align_to_f(frame_i0, 16)
                frame_i1, f_h_pad, f_w_pad = utils.align_to_f(frame_i1, 16)
                frame_i2, f_h_pad, f_w_pad = utils.align_to_f(frame_i2, 16)
                frame_i3, f_h_pad, f_w_pad = utils.align_to_f(frame_i3, 16)
                frame_i4, f_h_pad, f_w_pad = utils.align_to_f(frame_i4, 16)
                frame_i5, f_h_pad, f_w_pad = utils.align_to_f(frame_i5, 16)
                frame_i6, f_h_pad, f_w_pad = utils.align_to_f(frame_i6, 16)

                [b, c, h, w] = frame_i0.shape

                frame_input = torch.cat((frame_i0, frame_i1, frame_i2, frame_i3, frame_i4, frame_i5, frame_i6), 0)
                frame_input = frame_input.view(b, -1, c, h, w)
                time1 = time.time()
                frame_pred, _, _ = Net(frame_input)
                time2 = time.time()
                print(time2 - time1)
                frame_pred = frame_pred[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                
            fusion_frame_pred = utils.tensor2img(frame_pred)

            output_filename = os.path.join(output_dir, os.path.basename(frame_list[t]).replace(opts.file_suffix, '.jpg'))
            utils.save_img(fusion_frame_pred, output_filename)

    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" % (time_avg, len(times)))
