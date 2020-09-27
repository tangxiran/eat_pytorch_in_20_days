import os
import datetime
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
#打印时间
# def printbar():
#     nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print("\n"+"==\=="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# printbar()
def test_adaptiveMaxpool2d():
    # 自适应池化
    pool = nn.AdaptiveAvgPool2d((1,1))
    t =torch.randn(10,8,32,32)
    print(pool(t).shape)
    return pool(t).shape
# 定义模型
class Net(nn.Module):
    def __init__(self):
        #
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        # self.flatten = nn.Flatten()
        # self.flatten = torch.Tensor.reshape(-1,64)
        self.linear1 = nn.Linear(in_features=64,out_features= 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()
    # 前向计算过程
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.adaptive_pool(out)
        out = torch.Tensor.reshape(-1,64)
        # out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        y = self.sigmoid(out)
        return y
if __name__ == '__main__':
    import torch
    from torch import nn
    from torch.utils.data import Dataset,DataLoader
    from torchvision import transforms,datasets
    transform_train = transforms.Compose(
        [transforms.ToTensor()])
    transform_valid = transforms.Compose(
        [transforms.ToTensor()])
    ds_train = datasets.ImageFolder("./data/cifar2/train/",
                                    transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
    ds_valid = datasets.ImageFolder("./data/cifar2/test/",
                                    transform=transform_valid, target_transform=lambda t: torch.tensor([t]).float())

    # 打印种类，原始默认值文件夹的名字作为种类
    print(ds_train.class_to_idx)
    dl_train = DataLoader(ds_train,batch_size=50,shuffle=True,num_workers=0)
    dl_valid = DataLoader(ds_train,batch_size=50,shuffle=True,num_workers=0)

    # 查看部分样本
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 8))
    for i in range(9):
        img, label = ds_train[i] # 自带的 imgloader函数
        img = img.permute(1, 2, 0)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title("label = %d" % label.item())
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    # Pytorch的图片默认顺序是 Batch,Channel,Width,Height
    for x, y in dl_train:
        print('batch shape is ',x.shape, 'y shape is ',y.shape)
        break
    test_adaptiveMaxpool2d()
    net = Net()
    print(net)
    # 版本升级后再用
    # import torchkeras
    # print(torchkeras.summary(net , input_shape=(3,32,32)))
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    model = net
    model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.loss_func = torch.nn.BCELoss()
    model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
    model.metric_name = "auc"
