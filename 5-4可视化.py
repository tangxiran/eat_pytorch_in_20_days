
'''
我们主要介绍Pytorch中利用TensorBoard进行如下方面信息的可视化的方法。

可视化模型结构： writer.add_graph

可视化指标变化： writer.add_scalar

可视化参数分布： writer.add_histogram

可视化原始图像： writer.add_image 或 writer.add_images

可视化人工绘图： writer.add_figure
'''
import torch
print('version is ',torch.__version__)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model, summary
class Net(nn.Module):
    def __init__(self,inchannels,outclass):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, outclass)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        out = self.sigmoid(x)
        return out

if __name__ == '__main__':
    net =Net(inchannels=3,outclass=1)
    print(net)
    print( summary(net , input_shape=(3,32,32))
           )
    # writer = SummaryWriter('./data/tensorboard')
    # writer.add_graph(net, input_to_model=torch.rand(1, 3, 32, 32))
    # writer.close()

    from tensorboard import notebook


    # 等价于在命令行中执行 tensorboard --logdir ./data/tensorboard
    # 可以在浏览器中打开 http://localhost:6006/ 查看
    '''
    有时候在训练过程中，如果能够实时动态地查看loss和各种metric的变化曲线，那么无疑可以帮助我们更加直观地了解模型的训练情况。

    注意，writer.add_scalar仅能对标量的值的变化进行可视化。因此它一般用于对loss和metric的变化进行可视化分析
    '''
    # 二，可视化指标变化

    # 写入模型结构
    writer = SummaryWriter('./data/tensorboard')

    writer.add_graph(net,input_to_model=torch.rand(1,3,32,32))
    writer.close()
    # 查看模型结构
    from tensorboard import notebook
    #
    notebook.list()
    notebook.start("--logdir ./data/tensorboard")

    # 写入指标变化

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    # f(x) = a*x**2 + b*x + c的最小值
    x = torch.tensor(100.0, requires_grad=True)  # x需要被求导
    a = torch.tensor(1.0)
    b = torch.tensor(-2.0)
    c = torch.tensor(1.0)
    optimizer = torch.optim.SGD(params=[x], lr=0.01)


    def f(x):
        res = a * torch.pow(x, 2) + b * x + c
        return res
    for i in range(500): # i是第几次循环
        optimizer.zero_grad()
        y = f(x)
        y.backward()
        optimizer.step()
        writer.add_scalar("x", x.item(), i)  # 日志中记录x在第step i 的值
        writer.add_scalar("y", y.item(), i)  # 日志中记录y在第step i 的值
    writer.close()

    writer.close()
    print("y=",f(x).data,";","x=",x.data)

    # 查看启动的tensorboard程序
    #
    # 三，可视化参数分布
    # 如果需要对模型的参数(一般非标量)
    # 在训练过程中的变化进行可视化，可以使用
    # writer.add_histogram。
    #
    # 它能够观测张量值分布的直方图随训练步骤的变化趋势。

    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter


    # 创建正态分布的张量模拟参数矩阵
    def norm(mean, std):
        t = std * torch.randn((100, 20)) + mean
        return t


    writer = SummaryWriter('./data/tensorboard')
    for step, mean in enumerate(range(-10, 10, 1)):
        w = norm(mean, 1)
        writer.add_histogram("w", w, step)
        writer.flush()
    writer.close()
    # 可视化原始图像
    for i in range(1):

        import torch
        import torchvision
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, datasets

        transform_train = transforms.Compose(
            [transforms.ToTensor()])
        transform_valid = transforms.Compose(
            [transforms.ToTensor()])



        ds_train = datasets.ImageFolder("./data/cifar2/train/",
                                        transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
        ds_valid = datasets.ImageFolder("./data/cifar2/test/",
                                        transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
        print(ds_train.class_to_idx)

        dl_train = DataLoader(ds_train, batch_size=50, shuffle=True, num_workers=0)
        dl_valid = DataLoader(ds_valid, batch_size=50, shuffle=True, num_workers=0)

        dl_train_iter = iter(dl_train)
        images, labels = dl_train_iter.next()
        # 仅查看一张图片
        writer = SummaryWriter('./data/tensorboard')
        writer.add_image('images[0]', images[0])
        writer.close()

        # 将多张图片拼接成一张图片，中间用黑色网格分割
        writer = SummaryWriter('./data/tensorboard')
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)
        writer.add_image('image_grid', img_grid)
        writer.close()

        # 将多张图片直接写入
        writer = SummaryWriter('./data/tensorboard')
        writer.add_images("images", images, global_step=0)
        writer.close()

    for i in range(1):
        # 可视化人工绘图
        import torch
        import torchvision
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, datasets

        transform_train = transforms.Compose(
            [transforms.ToTensor()])
        transform_valid = transforms.Compose(
            [transforms.ToTensor()])

        ds_train = datasets.ImageFolder("./data/cifar2/train/",
                                        transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())
        ds_valid = datasets.ImageFolder("./data/cifar2/test/",
                                        transform=transform_train, target_transform=lambda t: torch.tensor([t]).float())

        print(ds_train.class_to_idx)
        from matplotlib import pyplot as plt

        figure = plt.figure(figsize=(8, 8))
        for i in range(9):
            img, label = ds_train[i]
            img = img.permute(1, 2, 0)
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(img.numpy())
            ax.set_title("label = %d" % label.item())
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        writer = SummaryWriter('./data/tensorboard')
        writer.add_figure('figure', figure, global_step=0)
        writer.close()


