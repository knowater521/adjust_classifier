#为保证原有数据安全，此脚本产生划分后的数据备份
#样本具有保序性

import os
import shutil
import random
from hyper_parameters import datadivision_ragne as division_range

import moxing as mox
mox.file.shift('os', 'mox')
'''
def data_divide(data_dir, train_dir, test_dir):
    for i in range(division_range):#数据集最后一项的数字，不是数据集总数
        imgpath = data_dir + "img_" + str(i) + ".jpg"
        txtpath = data_dir + "img_" + str(i) + ".txt"
        img_test_file = test_dir + "img_" + str(i) + ".jpg"
        txt_test_file = test_dir + "img_" + str(i) + ".txt"
        img_train_file = train_dir + "img_" + str(i) + ".jpg"
        txt_train_file = train_dir + "img_" + str(i) + ".txt" 
        if os.path.isfile(imgpath):
            if random.randint(0, 9) < 3:#30%概率数据选中为
                shutil.copy(imgpath, test_dir)
                shutil.copy(txtpath, test_dir)
                print("No." + str(i) + " has been divided into testset\n")
            else:
                shutil.copy(imgpath, train_dir)
                shutil.copy(txtpath, train_dir)
                print("No." + str(i) + " has been divided into trainset\n")

#if __name__ == "__main__":
    #data_divide(data_dir, train_dir, test_dir)#改成你系统里的对应目录
	#data_dir：原始数据
	#train_dir：训练集
	#test_dir：测试集
'''
def data_divide(data_dir, train_dir, test_dir):
    for i in range(1000):#数据集最后一项的数字，不是数据集总数
        imgpath = data_dir + "img_" + str(i) + ".jpg"
        txtpath = data_dir + "img_" + str(i) + ".txt"
        img_test_file = test_dir + "img_" + str(i) + ".jpg"
        txt_test_file = test_dir + "img_" + str(i) + ".txt"
        img_train_file = train_dir + "img_" + str(i) + ".jpg"
        txt_train_file = train_dir + "img_" + str(i) + ".txt"
        if mox.file.exists(imgpath):
            if random.randint(0, 9) < 3:#30%概率数据选中为
                mox.file.copy(imgpath, img_test_file)
                mox.file.copy(txtpath, txt_test_file)
                #shutil.copy(imgpath, test_dir)
                #shutil.copy(txtpath, test_dir)
                print("No." + str(i) + " has been divided into testset\n")
            else:
                mox.file.copy(imgpath, img_train_file)
                mox.file.copy(txtpath, txt_train_file)
                #shutil.copy(imgpath, train_dir)
                #shutil.copy(txtpath, train_dir)
                print("No." + str(i) + " has been divided into trainset\n")
