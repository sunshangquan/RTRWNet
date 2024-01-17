CUDA_VISIBLE_DEVICES=6 python test_video_rain.py -method P401_video_rain_self -epoch 3 -dataset Video_rain -task RainRemoval/original -data_dir data_NTU -model_name derain_self -checkpoint_dir ./checkpoints/P401_video_rain_self/ -list_filename ./lists/video_rain_removal_test.tx
/home/ssq/anaconda3/envs/torch14/bin/python test_video_rain.py -method single_stage2 -epoch 97 -dataset Video_rain -task self -data_dir /home/ssq/Desktop/phd/proj1/SPAC-SupplementaryMaterials/Extracted/Dataset_Testing_Synthetic/ -model_name single_stage2 -checkpoint_dir ./checkpoints/ -list_filename ./lists/video_rain_removal_test.txt
CUDA_VISIBLE_DEVICES=2 python test_video_rain.py -list_filename lists/nturain_test.txt -epoch 94 -data_dir /home1/ssq/proj1/evnet/data/image/Dataset_Testing_Synthetic/ -checkpoint_dir ./checkpoints/ -model_name nturain5_restormer_w_align_w_aggre_8
CUDA_VISIBLE_DEVICES=1 python test_video_rain.py -list_filename lists/dvd_test.txt -epoch 71 -data_dir /home1/ssq/data/DeepVideoDeblurring_Dataset/DVD10/ -checkpoint_dir ./checkpoints/ -model_name dvd_restormer_w_align_w_aggre_9/

CUDA_VISIBLE_DEVICES=0 python test_video_rain.py -list_filename lists/nturain_test.txt -epoch 45 -data_dir /home1/ssq/proj1/evnet/data/image/Dataset_Testing_Synthetic/ -checkpoint_dir ./checkpoints/ -model_name nturain5_rtrwnet_w_align_w_aggre_8
CUDA_VISIBLE_DEVICES=2 python test_video_rain.py -list_filename lists/nturain_test_real.txt -epoch 75 -data_dir /home1/ssq/proj1/evnet/data/others/ -checkpoint_dir ./checkpoints/ -model_name nturain5_rtrwnet_w_align_w_aggre_7

CUDA_VISIBLE_DEVICES=0 python test_video_rain.py -list_filename lists/rainsynlight25_test.txt -epoch 60 -data_dir /home1/ssq/data/RainSynLight25/video_rain_light/test/ -checkpoint_dir ./checkpoints/ -model_name rainsynlight25_rtrwnet_w_align_w_aggre_8 -file_suffix .png

CUDA_VISIBLE_DEVICES=0 python test_video_rain.py -list_filename lists/rainvidss_test.txt -epoch 55 -data_dir /home1/ssq/proj1/evnet/data/image/dataset_RainVIDSS/val/ -checkpoint_dir ./checkpoints/ -model_name rainvidss_rtrwnet_w_align_w_aggre_8
