# RTRWNet: Towards Real-Time and Real-World Video Rain Streak Removal

This repo is an implementation of "Towards Real-Time and Real-World Video Rain Streak Removal". 

## Installation

### Requirements:

- PyTorch 1.9
- CUDA 10.2
- natsort
- einops

For training, you need to install "correlation_package" for LiteFlowNet3

```
cd networks/correlation_package/
python setup.py install
```

## Data preparation

Please download the datasets of RainSynComplex25 and RainSynLight25 from [here](https://github.com/flyywh/J4RNet-Deep-Video-Deraining-CVPR-2018), the dataset of [NTURain](https://github.com/hotndy/SPAC-SupplementaryMaterials) and [RRODT](https://1drv.ms/f/s!AgRtHximCauqi0g3Yn7Qwytq5htN?e=N7u68t)

## Usage

### Testing

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py \
-list_filename lists/nturain_test.txt \
-epoch 75 \
-data_dir ./data/ \
-checkpoint_dir ./checkpoints/ \
-model_name nturain5_rtrwnet_w_align_w_aggre_7
```

### Training

Install "correlation_package" for LiteFlowNet3

```
cd networks/correlation_package/
python setup.py install
cd ../../
```

Change the path to dataset in "options/option_nturain.py". Then run

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_video_rain.py \
-checkpoint_dir ./checkpoints/ \
-crop_size 256 \
-model_name nturain5_rtrwnet_w_align_w_aggre_7 \
-lr_init 0.0001 \
-dataset_task nturain \
-ifAggregate 1 \
-batch_size 4
```

or

```
CUDA_VISIBLE_DEVICES=0 python train_video_rain.py \
-checkpoint_dir ./checkpoints/ \
-crop_size 256 \
-model_name nturain5_rtrwnet_w_align_w_aggre_7 \
-lr_init 0.0001 \
-dataset_task nturain \
-ifAggregate 1 \
-batch_size 1
```


### Demo Usage

Put the rainy frames in a [PATH]. Create a list file [LIST_FILE] in the format like 

```
FOLDER_RAINY_1
FOLDER_GT_1
FOLDER_RAINY_2
FOLDER_GT_2
...
```

If Ground-Truth video is absent in the case of real-world rainy video, "FOLDER_GT_1" is left empty like

```
FOLDER_RAINY_1

FOLDER_RAINY_2

...
```

Run

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py \
-list_filename [LIST_FILE] \
-epoch 75 \
-data_dir [PATH] \
-checkpoint_dir ./checkpoints/ \
-model_name nturain5_rtrwnet_w_align_w_aggre_7
```

Example:

```
CUDA_VISIBLE_DEVICES=0 python test_video_rain.py \
-list_filename lists/nturain_test_sub.txt \
-epoch 75 \
-data_dir ./data/ \
-checkpoint_dir ./checkpoints/ \
-model_name nturain5_rtrwnet_w_align_w_aggre_7
```

## Acknowledgments

This repository contains a PyTorch implementation of our approach RTRWNet based on [SLDNet](https://github.com/flyywh/CVPR-2020-Self-Rain-Removal), [Restormer](https://github.com/swz30/Restormer) and [LiteFlowNet3](https://github.com/lhao0301/pytorch-liteflownet3. 
