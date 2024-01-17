### python lib
import os, sys, math, random, glob, cv2
import numpy as np

### torch lib
import torch
import torch.utils.data as data

### custom lib
import utils
from natsort import natsorted

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]

class MultiFramesDataset(data.Dataset):

    def __init__(self, opts, mode):
        super(MultiFramesDataset, self).__init__()
        self.opts = opts
        self.mode = mode
        self.task_videos = []
        self.num_frames = []
        self.gt_videos = []
        self.T = 0
        list_filename = opts.list_filename if 'train' in mode else opts.test_list_filename
        self.frame_lists = []
        with open(list_filename) as f:
            videos = [line.rstrip() for line in f.readlines()]
            if 'paired_' in self.mode:
                videos_gt = videos[1::2]
                videos = videos[0::2]

        data_root = self.opts.data_dir if 'train' in mode else self.opts.datatest_dir
        self.data_root = data_root
        for video_index, video in enumerate(videos):
            self.task_videos.append(os.path.join(video))
            if 'paired_' in self.mode:
                if 'davis' in opts.dataset_task:
                    self.gt_videos.append(os.path.join(videos[video_index]))
                else:
                    self.gt_videos.append(os.path.join(videos_gt[video_index]))

            input_dir = os.path.join(data_root, video)
            tmp = []
            for suffix in ['.jpg', '.png']:
                tmp += glob.glob(os.path.join(input_dir, '*'+suffix))
            frame_list = natsorted([os.path.basename(file_) for file_ in tmp])
            self.frame_lists.append(frame_list)
            if len(frame_list) == 0:
                raise Exception("No frames in %s" %input_dir)


            self.num_frames.append(len(frame_list))

        print("[%s] Total %d videos (%d frames) for %s" %(self.__class__.__name__, len(self.task_videos), sum(self.num_frames), mode))

    def __len__(self):
        total_frame_num = np.sum(self.num_frames)
        drop_last_num = len(self.task_videos) * (self.opts.sample_frames-1)
        return total_frame_num - drop_last_num


    def __getitem__(self, index):
        # print("index", index)
        local_index = index
        video_index = 0
        while local_index > self.num_frames[video_index] - self.opts.sample_frames:
            local_index = local_index - (self.num_frames[video_index] - self.opts.sample_frames + 1)
            video_index += 1
        T = local_index

        video = self.task_videos[video_index]

        input_dir = os.path.join(self.data_root, video)

        frame_i = []
        frame_gt_i = []
        paried_index = []
        for t in range(T, T + self.opts.sample_frames):
            assert t < self.num_frames[video_index], "{}, {}, {}".format(t, T, self.num_frames[video_index])

            frame_i.append(utils.read_img(os.path.join(input_dir, self.frame_lists[video_index][t])))
            if 'paired_' in self.mode:
                if self.gt_videos[video_index] == '':
                    frame_gt = np.zeros_like(frame_i[-1])
                    paried_index.append(0)
                else:
                    frame_gt = utils.read_img(os.path.join(self.data_root, self.gt_videos[video_index], self.frame_lists[video_index][t]))
                    paried_index.append(1)
                frame_gt_i.append(frame_gt)
            else:
                paried_index.append(0)

        if self.mode == 'train':
            if self.opts.geometry_aug:

                H_in = frame_i[0].shape[0]
                W_in = frame_i[0].shape[1]

                sc = np.random.uniform(self.opts.scale_min, self.opts.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be greater than opts.crop_size
                if H_out < W_out:
                    if H_out < self.opts.crop_size:
                        H_out = self.opts.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.opts.crop_size:
                        W_out = self.opts.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.opts.sample_frames):
                    frame_i[t] = cv2.resize(frame_i[t], (W_out, H_out))

            cropper = RandomCrop(frame_i[0].shape[:2], (self.opts.crop_size, self.opts.crop_size))
            
            for t in range(self.opts.sample_frames):
                frame_i[t] = cropper(frame_i[t])

            if self.opts.geometry_aug:
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.opts.sample_frames):
                        frame_i[t] = cv2.flip(frame_i[t], flipCode=0)

            if self.opts.order_aug:
                ## reverse temporal order
                if np.random.random() >= 0.5:
                    frame_i.reverse()
            data = []
            for t in range(self.opts.sample_frames):
                data.append(torch.from_numpy(frame_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())

            return data

        elif self.mode == 'paired_train':
            if self.opts.geometry_aug:

                H_in = frame_i[0].shape[0]
                W_in = frame_i[0].shape[1]

                sc = np.random.uniform(self.opts.scale_min, self.opts.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be greater than opts.crop_size
                if H_out < W_out:
                    if H_out < self.opts.crop_size:
                        H_out = self.opts.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.opts.crop_size:
                        W_out = self.opts.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.opts.sample_frames):
                    frame_i[t] = cv2.resize(frame_i[t], (W_out, H_out))
                    frame_gt_i[t] = cv2.resize(frame_gt_i[t], (W_out, H_out))

            cropper = RandomCrop(frame_i[0].shape[:2], (self.opts.crop_size, self.opts.crop_size))
            
            for t in range(self.opts.sample_frames):
                tmp = np.concatenate([frame_i[t], frame_gt_i[t]], 2)
                tmp = cropper(tmp)
                frame_i[t], frame_gt_i[t] = tmp[...,:3], tmp[...,3:]

            if self.opts.geometry_aug:
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.opts.sample_frames):
                        frame_i[t] = cv2.flip(frame_i[t], flipCode=0)
                        frame_gt_i[t] = cv2.flip(frame_gt_i[t], flipCode=0)

            if self.opts.order_aug:
                ## reverse temporal order
                if np.random.random() >= 0.5:
                    frame_i.reverse()
                    frame_gt_i.reverse()
            data = []
            gts = []
            for t in range(self.opts.sample_frames):
                if 'davis' in self.opts.dataset_task:
                    gt_img = torch.from_numpy(frame_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous()
                    noise_level = torch.ones((1,1,1)) * 50/255.
                    noise = torch.normal(mean=0, std=noise_level.expand_as(gt_img))
                    data.append(gt_img + noise)
                else:
                    data.append(torch.from_numpy(frame_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
                gts.append(torch.from_numpy(frame_gt_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
            return data, gts, paried_index
        elif self.mode == "paired_test":
            ## resize image to avoid size mismatch after downsampline and upsampling
            H_i = frame_i[0].shape[0]
            W_i = frame_i[0].shape[1]

            H_o = int(float(H_i) // self.opts.size_multiplier * self.opts.size_multiplier)
            W_o = int(float(W_i) // self.opts.size_multiplier * self.opts.size_multiplier)

            for t in range(self.opts.sample_frames):
                frame_i[t] = cv2.resize(frame_i[t], (W_o, H_o))
                frame_gt_i[t] = cv2.resize(frame_gt_i[t], (W_o, H_o))
            data = []
            gts = []
            for t in range(self.opts.sample_frames):
                data.append(torch.from_numpy(frame_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
                if 'davis' in self.opts.dataset_task:
                    gt_img = data[-1]
                    noise_level = torch.ones((1,1,1)) * 50/255.
                    noise = torch.normal(mean=0, std=noise_level.expand_as(gt_img))
                    data[-1] = data[-1] + noise
                gts.append(torch.from_numpy(frame_gt_i[t].transpose(2, 0, 1).astype(np.float32)).contiguous())

            return data, gts, self.task_videos[video_index]
        else:
            raise Exception("Unknown mode (%s)" %self.mode)

        ### convert (H, W, C) array to (C, H, W) tensor
        
