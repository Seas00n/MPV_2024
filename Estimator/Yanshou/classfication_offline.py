import numpy as np
import cv2
from PIL import Image
from Environment.Environment import *
import os
import sys
import random
import torch

sys.path.append("//")

str = input("按回车统计")
path_to_IMAGE = "/media/yuxuan/My Passport/testClassification/IMAGE/"

num_each_kind = 200
level_file_list = []
upslope_file_list = []
downslope_file_list = []
upstair_file_list = []
downstair_file_list = []
for env_ in ["0/", "1/", "2/", "3/", "4/"]:
    path_ = path_to_IMAGE + env_
    f_list = os.listdir(path_)
    all_idx = range(20, len(f_list))
    rand_idx = random.sample(all_idx, num_each_kind)
    if env_ == "0/":
        for idx in rand_idx:
            level_file_list.append(path_ + f_list[idx])
    elif env_ == "1/":
        for idx in rand_idx:
            upslope_file_list.append(path_ + f_list[idx])
    elif env_ == "2/":
        for idx in rand_idx:
            downslope_file_list.append(path_ + f_list[idx])
    elif env_ == "3/":
        for idx in rand_idx:
            upstair_file_list.append(path_ + f_list[idx])
    elif env_ == "4/":
        for idx in rand_idx:
            downstair_file_list.append(path_ + f_list[idx])

model = torch.load('/home/yuxuan/Project/CCH_Model/realworld_model_epoch_29.pt',
                   map_location=torch.device('cpu'))

num_correct = 0
true_pred = 0
for file in level_file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = img.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    pred = int(pred[:])
    if pred == true_pred:
        num_correct += 1
    else:
        print("分类错误：{}".format(file))
        print("被错误分类为：{}".format(Env_Type(pred)))
num_correct_level = num_correct
print("平地分类正确率:{}%".format(num_correct / 200 * 100))

num_correct = 0
true_pred = 1
for file in upslope_file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = img.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    pred = int(pred[:])
    if pred == true_pred:
        num_correct += 1
    else:
        print("分类错误：{}".format(file))
        print("被错误分类为：{}".format(Env_Type(pred)))
num_correct_upslope = num_correct
print("上楼分类正确率:{}%".format(num_correct / 200 * 100))

num_correct = 0
true_pred = 2
for file in downslope_file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = img.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    pred = int(pred[:])
    if pred == true_pred:
        num_correct += 1
    else:
        print("分类错误：{}".format(file))
        print("被错误分类为：{}".format(Env_Type(pred)))
num_correct_upslope = num_correct
print("下楼分类正确率:{}%".format(num_correct / 200 * 100))

num_correct = 0
true_pred = 3
for file in upstair_file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = img.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    pred = int(pred[:])
    if pred == true_pred:
        num_correct += 1
    else:
        print("分类错误：{}".format(file))
        print("被错误分类为：{}".format(Env_Type(pred)))
num_correct_upslope = num_correct
print("上坡分类正确率:{}%".format(num_correct / 200 * 100))

num_correct = 0
true_pred = 4
for file in downstair_file_list:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_input = img.reshape(1, 1, 100, 100).astype('uint8')
    img_input = torch.tensor(img_input, dtype=torch.float)
    with torch.no_grad():
        output = model(img_input)
    pred = output.data.max(1)[1].cpu().numpy()
    pred = int(pred[:])
    if pred == true_pred:
        num_correct += 1
    else:
        print("分类错误：{}".format(file))
        print("被错误分类为：{}".format(Env_Type(pred)))
num_correct_upslope = num_correct
print("下坡分类正确率:{}%".format(num_correct / 200 * 100))