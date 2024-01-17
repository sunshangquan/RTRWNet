#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np
import multiprocessing as mp

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

### custom lib
import datasets_multiple
import utils
from utils import *
import torch.nn.init as init
from torch.nn.init import *
import torchvision
from loss import *

import torch.nn.functional as F

from networks import LiteFlowNet
from networks.SoftMedian import softMedian, softMin, softMax
from networks.TDModel import TDModel as Model
from networks.rtrwnet import Model
import time
import os

import faulthandler
def EPE_loss(inp, gt):
    tmp = torch.norm(gt - inp, 2, 1)
    return tmp.mean()
def rgb_2_Y(im):
    b, c, h, w = im.shape
    weight = torch.Tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(im)
    return (weight * im).mean(1, keepdim=True).repeat(1, c, 1, 1)

def seed_torch(seed=3407):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

# 在import之后直接添加以下启用代码即可
faulthandler.enable()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fast Blind Video Temporal Consistency")
    parser.add_argument('-Net',   type=str,     default="mymodel",     help='Multi-frame models for hanlde videos')

    parser.add_argument('-nf',              type=int,     default=16,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=2,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')

    parser.add_argument('-dataset_task',  type=str,     default='nturain', choices=['nturain', 'rainvidss', 'rainsynlight25', 'rainsyncomplex25'],    help='dataset-task pairs list')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=32,               help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.4,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=7,                help='#frames for training')
        
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L2",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')

    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=25,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.01,             help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=16,               help='number of threads for data loader to use')
    parser.add_argument('-batch_size',      type=int,     default=16,               help='size of batch for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='.jpg',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')

    parser.add_argument('-list_filename',   type=str,      help='use cpu?')
    parser.add_argument('-test_list_filename',   type=str,      help='use cpu?')

    parser.add_argument('-ifAlign',             type=int,     default=1,                help='if align or not')
    parser.add_argument('-ifAggregate',             type=int,     default=1,                help='if aggregate or not')

    opts = parser.parse_args()
    if opts.dataset_task == 'nturain':
        from options.option_nturain import *
    elif opts.dataset_task == 'rainvidss':
        from options.option_rainvidss import * 
    elif opts.dataset_task == 'rainsynlight25':
        from options.option_rainsynlight25 import *  
    elif opts.dataset_task == 'rainsyncomplex25':
        from options.option_rainsyncomplex25 import *

    opts.checkpoint_dir = opt.checkpoint_dir
    opts.data_dir = opt.data_dir
    opts.datatest_dir = opt.datatest_dir

    opts.list_filename = opt.list_filename
    opts.test_list_filename = opt.test_list_filename
#    opts.model_name = opt.model_name
#    opts.batch_size = opt.batch_size
    opts.suffix = opt.suffix

    opts.train_epoch_size = opt.train_epoch_size
    opts.valid_epoch_size = opt.valid_epoch_size
    opts.epoch_max = opt.epoch_max
    opts.threads = opt.threads
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m   

    opts.size_multiplier = 2 ** 4
    print(opts)

    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)

    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    print('===> Initializing model from %s...' %opts.Net)
    Net = Model(opts)#networks.mymodel2.Model()
    Net.flow_estimator = torch.nn.DataParallel(Net.flow_estimator)
    Net.frame_restorer = torch.nn.DataParallel(Net.frame_restorer)
    Net.epoch = 0 
#    Net.apply(weight_init)
    
    ### Load pretrained FlowNet2
    opts.rgb_max = 1.0
    opts.fp16 = False

    FlowNet = LiteFlowNet.LightFlowNet()
    model_filename = os.path.join("./pretrained_models", "network-sintel.pytorch")
    print("===> Load %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet.load_state_dict(checkpoint)

    if opts.solver == 'SGD':
        optimizer = optim.SGD(Net.parameters(), \
                              lr=opts.lr_init, momentum=opts.momentum, weight_decay= opts.weight_decay )
    elif opts.solver == 'ADAM':
        optimizer = optim.AdamW([ \
                                {'params': Net.frame_restorer.parameters(), 'lr': opts.lr_init }, \
                               ], lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

        optimizer_flow = optim.AdamW([ \
                                {'params': Net.flow_estimator.parameters(), 'lr': 1e-4 }, \
                               ], \
                               lr=1e-4, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))

        lr_scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_flow, mode="min", factor=0.5, patience=6, threshold=1e-6)

    else:
        raise Exception("Not supported solver (%s)" %opts.solver)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, threshold=1e-6)


    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = -1

    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    if epoch_st >= 0:
        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')
        Net, FlowNet, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow = utils.load_model(Net, FlowNet, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow, opts, epoch_st)

    print(Net)

    num_params = utils.count_network_parameters(Net)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')

    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)

    device = torch.device(0 if opts.cuda else "cpu")

    Net.frame_restorer = Net.frame_restorer.cuda(device=device)
    Net.flow_estimator = Net.flow_estimator.cuda(device=device)
    FlowNet = FlowNet.cuda(device=device)

    Net.train()
    FlowNet.eval()

    train_dataset = datasets_multiple.MultiFramesDataset(opts, "paired_train")
    if opts.test_list_filename:
        val_dataset = datasets_multiple.MultiFramesDataset(opts, "paired_test")
        val_loader = utils.create_data_loader(val_dataset, opts, "paired_test")
        valid_dir = os.path.join(opts.model_dir, 'validation')
        os.makedirs(os.path.join(valid_dir), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'pred'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'input'), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, 'diff'), exist_ok=True)


    loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
    loss_fn2 = torch.nn.MSELoss(reduce=True, size_average=True)
    PSNRs = [0]
    while Net.epoch < opts.epoch_max:
        Net.epoch += 1

        data_loader = utils.create_data_loader(train_dataset, opts, "paired_train")
#        current_lr = utils.learning_rate_decay(opts, Net.epoch)

#        for param_group in optimizer.param_groups:
#                param_group['lr'] = current_lr
#        for param_group in optimizer_flow.param_groups:
#                param_group['lr'] = current_lr*0.001
        
        prev_epoch_distill_loss = 1e10
        distill_loss_list = []
        overall_loss_list = []
        ts = datetime.now()
        
        for iteration, batch in enumerate(data_loader, 1):
            batch, gt, paired_index = batch
            print("paired_index", paired_index)
            unpaired_index = np.where(np.array(paired_index[0])==0)[0]
            paired_index = np.where(np.array(paired_index[0])==1)[0]

            # if iteration >= 10:
            #     break
            total_iter = (Net.epoch - 1) * opts.train_epoch_size + iteration
            cross_num = 1

            frame_i = []

            for t in range(opts.sample_frames):
                frame_i.append(batch[t * cross_num].cuda(device=device))

            data_time = datetime.now() - ts
            ts = datetime.now()

            optimizer.zero_grad()
            optimizer_flow.zero_grad()

            [b, c, h, w] = frame_i[0].shape

            frame_i0, \
            frame_i1, \
            frame_i2, \
            frame_i3, \
            frame_i4, \
            frame_i5, \
            frame_i6 = frame_i

            
            flow_warping = networks.LiteFlowNet.backwarp

            seq_around = (frame_i0,
                          frame_i1,
                          frame_i2,
                          frame_i4,
                          frame_i5,
                          frame_i6)

            num_around = len(seq_around)           
            
            ###### Train Derain ######

            seq_input = torch.cat(frame_i, 1).view(b, -1, c, h, w)

            gt_tensor = torch.cat(gt, 1).view(b, -1, c, h, w).cuda(device=device)

            frame_target = gt_tensor[:,3]

            distill_loss = torch.Tensor([0.0]).float().cuda(device=device)
            flow_inward = [FlowNet(frame_i3, frame) for i, frame in enumerate(seq_around)] # teacher on lq
            flow_inward_tensor = torch.cat(flow_inward, 1).detach()
                
            if opts.ifAlign and len(paired_index) != 0:
                flow_list = Net.predict_flow(seq_input.contiguous().view(b, -1, h, w)[paired_index], Net.scales) # student on lq
#                flow_list_gt = Net.predict_flow(gt_tensor.contiguous().view(b, -1, h, w), Net.scales) # student on gt
#                flow_inward = [FlowNet(frame_i3, frame) for i, frame in enumerate(seq_around)] # teacher on lq
#                flow_inward_tensor = torch.cat(flow_inward, 1).detach()
                flow_inward_gt = [FlowNet(gt_tensor[paired_index,3], gt_tensor[paired_index,i]) for i in range(7) if i != 3] # teacher on gt
                flow_inward_gt_tensor = torch.cat(flow_inward_gt,1).detach()

                for j in range(len(flow_list)):
#                    flow_tea = F.interpolate(flow_inward_tensor, scale_factor=1./Net.scales[j], mode="bilinear", align_corners=False)
#                    flow = flow_list[j]
#                    distill_loss += EPE_loss(flow, flow_tea)
 
#                    flow_tea_gt = F.interpolate(flow_inward_gt_tensor, scale_factor=1./Net.scales[j], mode="bilinear", align_corners=False)
#                    flow = flow_list_gt[j]
#                    distill_loss += EPE_loss(flow, flow_tea_gt)

                    flow_tea_gt = F.interpolate(flow_inward_gt_tensor, scale_factor=1./Net.scales[j], mode="bilinear", align_corners=False)
                    flow = flow_list[j]
                    distill_loss += EPE_loss(flow, flow_tea_gt)

                distill_loss /= len(flow_list) #* 3# * num_around * 1000

                distill_loss.backward()            
                optimizer_flow.step()

            ################ Reconstruction Loss ################
            frames_warp = [flow_warping(frame, flow_inward_tensor[:,2*i:2*i+2]).unsqueeze(0) for i, frame in enumerate(seq_around)] + [frame_i3.unsqueeze(0), ]
            center_median = torch.cat((frames_warp), 0).median(0)[0].detach()
            overall_loss = torch.Tensor([0.0]).float().cuda(device=device)
            if True: #Net.epoch > 1 and prev_epoch_distill_loss <= 0.01:
                frame_pred, _, seq_input_warp = Net(seq_input, ifInferece=False, )#flow_tea=flow_inward_gt_tensor if opts.ifAlign else None)
#                center_median = seq_input_warp.detach().view(b,-1,c,h,w)[:,3:-3].median(1)[0]
                print(paired_index, unpaired_index, center_median.shape, frame_pred.shape, seq_input_warp.shape)
                loss1, loss2, loss3 = torch.Tensor([0.0]).float().cuda(device=device), torch.Tensor([0.0]).float().cuda(device=device), torch.Tensor([0.0]).float().cuda(device=device)
                if len(paired_index) != 0:
                    loss1 = loss_fn(frame_pred[paired_index], frame_target[paired_index])
                if len(unpaired_index) != 0:
                    loss2 += loss_center_median(frame_pred[unpaired_index], center_median[unpaired_index], frame_i3[unpaired_index])
                    
#                    PSNR_prev = np.mean(PSNRs)
#                    if PSNR_prev >= 35:
#                        loss2 += loss_fn(frame_pred[unpaired_index], center_median[unpaired_index]) 
#                    else:
#                        loss2 += torch.Tensor([0.0]).float().cuda(device=device)
                flow_outward = [FlowNet(frame, frame_pred) for i, frame in enumerate(seq_around)] # teacher on lq
                frames_out_warp = [flow_warping(frame_pred, flow) for i, flow in enumerate(flow_outward)]
#                pool = mp.Pool()
#                frames_out_warp = pool.starmap(flow_warping, list(zip([frame_pred,]*len(flow_outward), flow_outward)))
                for frame1, frame2 in zip(seq_around, frames_out_warp):
                    loss3 += loss_fn(frame2, frame1)
                overall_loss = loss1 + 0.1 * loss2 + loss3 / 60

                total_loss = overall_loss #+ distill_loss # + overall_loss + optical_loss
                total_loss.backward()
                optimizer.step()

            with torch.no_grad():
                save_img = torchvision.utils.make_grid(torch.cat([gt_tensor.contiguous().view(-1, c, h, w), seq_input.contiguous().view(-1, c, h, w)], 0), nrow=7)
                torchvision.utils.save_image(save_img, os.path.join(valid_dir, "target.jpg"))
                torchvision.utils.save_image(torch.cat([frame_pred, seq_input_warp.contiguous().view(-1, c, h, w), frame_target],0), os.path.join(valid_dir, "seq_input.jpg"))
                torchvision.utils.save_image(center_median, os.path.join(valid_dir, "center_median.jpg"))

            overall_loss_list.append(overall_loss.item())
            distill_loss_list.append(distill_loss.item())

            network_time = datetime.now() - ts

            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(Net.epoch, iteration, len(data_loader))

            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            info += "\tmodel = %s\n" %opts.model_name

            loss_writer.add_scalar('Rect Loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Rect Loss", overall_loss.item())

            if not isinstance(distill_loss, int):
                distill_loss = distill_loss.item()
            loss_writer.add_scalar('Distill Loss', distill_loss, total_iter)
            info += "\t\t%25s = %f\n" %("Distill Loss", distill_loss)

            print(info)

        utils.save_model(Net, FlowNet, optimizer, optimizer_flow, lr_scheduler, lr_scheduler_flow, opts)        
        lr_scheduler.step(np.mean(overall_loss_list))
        prev_epoch_distill_loss = np.mean(distill_loss_list)
        lr_scheduler_flow.step(prev_epoch_distill_loss)

        ################################# test #################################
        if opts.test_list_filename:
            PSNRs = []
            times = []
            for iteration, (batch, batch_gt, file_name) in enumerate(val_loader, 1):
#                if iteration >= 10:
#                    break
                print(file_name[0])
                file_name = file_name[0].replace("../", "")
                os.makedirs(os.path.join(valid_dir, 'gt', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'input', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'pred', file_name), exist_ok=True)
                os.makedirs(os.path.join(valid_dir, 'diff', file_name), exist_ok=True)

                frame_gt = batch_gt[3]
                with torch.no_grad():
                    batch_ = []
                    for frame in batch:
                        frame = frame.cuda(device=device)
                        frame, f_h_pad, f_w_pad = utils.align_to_f(frame, opts.size_multiplier)
                        batch_.append(frame)

                    frame_i0, \
                    frame_i1, \
                    frame_i2, \
                    frame_i3, \
                    frame_i4, \
                    frame_i5, \
                    frame_i6 = batch_

                    [b, c, h, w] = frame_i0.shape

                    flow_warping = networks.LiteFlowNet.backwarp
                    seq_around = (frame_i0,
                                  frame_i1,
                                  frame_i2,
                                  frame_i4,
                                  frame_i5,
                                  frame_i6)

                    num_around = len(seq_around)
                    seq_input = torch.cat(batch_, 1).view(b, -1, c, h, w)
                    # seq_input[:,:,3] = softMedian(torch.cat([frame.unsqueeze(2) for frame in warp_inward]+[frame_i3.unsqueeze(2)], 2), 2) 
                    time1 = time.time()
                    frame_pred, _, _ = Net(seq_input)
                    time2 = time.time()
                    used_time = time2-time1
                    times.append(used_time)
                    frame_pred = frame_pred[:, :, 0:h-f_h_pad, 0:w-f_w_pad]
                print("Time:", used_time)
                fusion_frame_pred = utils.tensor2img(frame_pred)

                output_filename = os.path.join(valid_dir, 'pred', file_name, "%05d.jpg" % iteration)
                utils.save_img(fusion_frame_pred, output_filename)

                frame_input = utils.tensor2img(frame_i3.view(b, c, h, w))
                output_filename = os.path.join(valid_dir, 'input', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_input, output_filename)

                # frame_median = utils.tensor2img(center_median.view(1, c, h, w))
                # output_filename = os.path.join(valid_dir, "%05d_median.jpg" % iteration)
                # utils.save_img(frame_median, output_filename)

                # frame_min = utils.tensor2img(center_min.view(1, c, h, w))
                # output_filename = os.path.join(valid_dir, "%05d_min.jpg" % iteration)
                # utils.save_img(frame_min, output_filename)

                # frame_max = utils.tensor2img(center_max.view(1, c, h, w))
                # output_filename = os.path.join(valid_dir, "%05d_max.jpg" % iteration)
                # utils.save_img(frame_max, output_filename)

                frame_gt = utils.tensor2img(frame_gt.view(b, c, h, w))
                output_filename = os.path.join(valid_dir, 'gt', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_gt, output_filename)

                frame_diff = utils.tensor2img((frame_i3).view(b, c, h, w)) - frame_gt
                output_filename = os.path.join(valid_dir, 'diff', file_name, "%05d.jpg" % iteration)
                utils.save_img(frame_diff, output_filename)
                if iteration >= 110:
                    continue
                psnr = utils.compute_psnr(frame_gt, fusion_frame_pred, MAX=1.)
                print(psnr)
                PSNRs.append(psnr)
            loss_writer.add_scalar('Epoch PSNR', np.mean(PSNRs), Net.epoch)
            lr_cur = optimizer.state_dict()['param_groups'][0]['lr']
            loss_writer.add_scalar('Epoch LR', lr_cur, Net.epoch)
            lr_flow_cur = optimizer_flow.state_dict()['param_groups'][0]['lr']
            loss_writer.add_scalar('Epoch Flow LR', lr_flow_cur, Net.epoch)

            loss_writer.add_scalar('Epoch overall loss', np.mean(np.nan_to_num(overall_loss_list)), Net.epoch)
            loss_writer.add_scalar('Epoch distill loss', np.mean(np.nan_to_num(distill_loss_list)), Net.epoch)
            print("Used Time:", np.mean(times))
            with open(os.path.join(valid_dir, "psnr_val.txt"), "a+") as f_:
                for iteration, psnr in enumerate(PSNRs):
                    f_.write("{}: {}\n".format(iteration, psnr))
                f_.write("Average: {}\n".format(np.mean(PSNRs)))
             
