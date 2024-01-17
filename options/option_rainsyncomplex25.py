
class item:
    def __init__(self):
        self.name = ''

opt = item()

opt.checkpoint_dir = './checkpoints/'
opt.data_dir = '/home1/ssq/data/RainSynComplex25/video_rain_heavy/train/'
opt.datatest_dir = '/home1/ssq/data/RainSynComplex25/video_rain_heavy/test/'

opt.list_filename = './lists/rainsyncomplex25_train.txt' 
opt.test_list_filename = './lists/rainsyncomplex25_test_sub.txt'
#opt.test_list_filename = './lists/nturain_test.txt'

opt.self_tag = 'video_rain_self'

opt.model_name = 'single_stage_resnet_w_softMedian'
opt.batch_size = 24

opt.threads = 16
opt.input_show = False
opt.suffix = '.png'

opt.train_epoch_size = 500
opt.valid_epoch_size = 20
opt.epoch_max = 100
