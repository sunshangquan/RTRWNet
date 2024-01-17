CUDA_VISIBLE_DEVICES=1 /home/ssq/anaconda3/bin/python train_video_rain.py -checkpoint_dir ./checkpoints/P202_video_rain_pretrain_formal_code/ -data_dir Video_rain -list_filename ./lists/video_rain_removal_train.txt -crop_size 64
/home/ssq/anaconda3/envs/torch14/bin/python train_video_rain.py -checkpoint_dir ./checkpoints/ -list_filename ./lists/video_rain_removal_train.txt -crop_size 128
/home/ssq/anaconda3/envs/torch14/bin/python train_video_rain.py -checkpoint_dir ./checkpoints/ -list_filename ./lists/video_rain_removal_train.txt -crop_size 128
CUDA_VISIBLE_DEVICES=0,1,3 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 64 -model_name nturain5_restormer_w_align_wo_aggre -lr_init 0.00001 -dataset_task nturain -ifAggregate 0
CUDA_VISIBLE_DEVICES=0,1,3 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 64 -model_name nturain5_restormer_w_align_wo_aggre -lr_init 0.00001 -dataset_task nturain -ifAggregate 0
CUDA_VISIBLE_DEVICES=0,1,3 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 64 -model_name nturain5_restormer_w_align_wo_aggre -lr_init 0.00001 -dataset_task nturain -ifAggregate 0
CUDA_VISIBLE_DEVICES=0,1,3 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 64 -model_name nturain5_restormer_w_align_wo_aggre -lr_init 0.00001 -dataset_task nturain -ifAggregate 0
CUDA_VISIBLE_DEVICES=0,1,2 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 128 -model_name nturain5_restormer_w_align_w_aggre_8 -lr_init 0.0001 -dataset_task nturain -ifAggregate 1 -batch_size 8
CUDA_VISIBLE_DEVICES=0,1,2 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 128 -model_name dvd_restormer_w_align_w_aggre_8 -lr_init 0.0001 -dataset_task dvd -ifAggregate 1 -batch_size 12

CUDA_VISIBLE_DEVICES=1,2,3,0 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 256 -model_name nturain5_rtrwnet_w_align_w_aggre_7 -lr_init 0.0001 -dataset_task nturain -ifAggregate 1 -batch_size 4
CUDA_VISIBLE_DEVICES=1,2,3,0 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 256 -model_name rainvidss_rtrwnet_w_align_w_aggre_8 -lr_init 0.0001 -dataset_task rainvidss -ifAggregate 1 -batch_size 4
CUDA_VISIBLE_DEVICES=1,2,3,0 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 256 -model_name rainsyncomplex25_rtrwnet_w_align_w_aggre_8 -lr_init 0.0001 -dataset_task rainsyncomplex25 -ifAggregate 1 -batch_size 4
CUDA_VISIBLE_DEVICES=1,2,3,0 python train_video_rain.py -checkpoint_dir ./checkpoints/ -crop_size 256 -model_name rainsynlight25_rtrwnet_w_align_w_aggre_8 -lr_init 0.0001 -dataset_task rainsynlight25 -ifAggregate 1 -batch_size 4



