
class item:
    def __init__(self):
        self.name = ''

opt = item()

opt.checkpoint_dir = './checkpoints/'
opt.data_dir = '/home/ssq/Desktop/phd/proj1/SPAC-SupplementaryMaterials/Extracted/Dataset_Training_Synthetic/' 
opt.datatest_dir = '/home/ssq/Desktop/phd/proj1/SPAC-SupplementaryMaterials/Extracted/Dataset_Testing_Synthetic/' 


opt.list_filename = './lists/nturain_train_semi.txt'
opt.list_filename = './lists/nturain_train_semi_rrodt.txt'
# opt.test_list_filename = './lists/nturain_test_sub.txt'
opt.test_list_filename = './lists/nturain_test.txt'

opt.self_tag = 'video_rain_self'

opt.model_name = 'single_stage_resnet_w_softMedian'
opt.batch_size = 24

opt.threads = 16
opt.input_show = False
opt.suffix = '.jpg'

opt.train_epoch_size = 500
opt.valid_epoch_size = 20
opt.epoch_max = 100
