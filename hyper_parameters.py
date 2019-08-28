data_dir = 'E:/分类项目/classifier_try/dataset/'
train_dir = 'E:/分类项目/classifier_try/trainset/'
test_dir = 'E:/分类项目/classifier_try/testset/'
log_dir = 'E:/分类项目/classifier_try/log/'
save_trained_net = 'E:/分类项目/classifier_try/trained_net.pkl'
save_trained_net_params = 'E:/分类项目/classifier_try/trained_net_params.pkl'

model_num = 101
epochs = 5
lr_coefficient = 5
weight_decay = 1e-8
batch_size=64

dividing = 0
datadivision_ragne = 1000

model_dir = 'E:/分类项目/classifier_try/mod/'

#数据增广参数
img_size = 224
cro_size = 224
angle = 45
brightness=0.05
contrast=0.1 
saturation=0.3 
hue=0.2