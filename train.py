import torch as t
from torch.nn import functional as F
from torch.autograd import Variable
from draw_and_save import curve_draw
from dloss import dloss
from ListToTensor import ListToTensor
from dataenhance import *
from ResNet import *
from dataloader import *
from train_once import *
from evaluate import *
from hyper_parameters import *
from datadivision import data_divide
import moxing as mox
mox.file.shift('os', 'mox')
#from ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

def train(epochs=120, 
          init_lr=init_lr, 
          weight_decay=weight_decay, 
          model_num=model_num, 
          batch_size=batch_size, 
          train_dir=train_dir, 
          test_dir=test_dir, 
          log_dir=log_dir, 
          ):
    #divide data
    if dividing == 1:
        data_divide(data_dir, train_dir, test_dir)
    #division
    print("data loading......\n")
    transform = enhance_transform()
    transform_std = transform_standard()
    trainset = DataClassify(train_dir, transforms=transform)
    testset = DataClassify(test_dir, transforms=transform_std)
    train_length = len(trainset)
    test_length = len(testset)
    data_loader_train = t.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    data_loader_test = t.utils.data.DataLoader(testset, batch_size, shuffle=False)
    print("loading complete\n")

    if model_num==0:
        exit(0)
    elif model_num==18:
        net = resnet18()
    elif model_num==34:
        net = resnet34()
    elif model_num==50:
        net = resnet50()
    elif model_num==101:
        net = resnet101()
    elif model_num==152:
        net = resnet152()

    #确定网络基于cpu还是gpu
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net.to(device)


    cost = t.nn.CrossEntropyLoss()
    train_loss_list = []
    train_accurate_list = []
    test_loss_list = []
    test_accurate_list = []

    for epoch in range(epochs):
        print("epoch " + str(epoch+1) + " start training...\n")
        net.train()
        learning_rate = dloss(train_loss_list, init_lr, lr_coefficient, init_lr)
        optimizer = t.optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)

        run_loss, corr = train_once(data_loader_train, net, optimizer, cost)
        train_loss_list.append(run_loss/train_length)
        train_accurate_list.append(corr/train_length)

        print('epoch %d, training loss %.6f, training accuracy %.4f ------\n' %(epoch+1, run_loss/train_length, corr/train_length))
        print("epoch " + str(epoch+1) + " finish training\n")
        print("-----------------------------------------------\n")
        print("epoch " + str(epoch+1) + " start testing...\n")

        net.eval()
        test_corr = evaluate(net, data_loader_test)
        test_accurate_list.append(test_corr/test_length)
        print('epoch %d, testing accuracy %.4f ------\n' %(epoch+1, test_corr/test_length))
        print("epoch " + str(epoch+1) + " finish testing\n")
        print("-----------------------------------------------\n")

    t.save(net, save_trained_net)

    t.save(net.state_dict(), save_trained_net_params)

    curve_draw(train_loss_list, train_accurate_list, test_accurate_list, log_dir)

    print("mission complete")


if __name__ == '__main__':
    train(epochs=epochs,
           model_num=model_num,
           train_dir=train_dir, 
           test_dir=test_dir,
           log_dir=log_dir)