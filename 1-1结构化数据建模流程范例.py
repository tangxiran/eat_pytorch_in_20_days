import os
import  datetime
# def printTime():
#     # 打印时间
#     import datetime
#     nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print("\n" + "time is :"+ "%s" % nowtime)
def processData(dfdata):
    # data preProcess
    dfresult =pd.DataFrame()
    # pclass
    dfPclass  =pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    # axis= 1 按列拼接
    dfresult = pd.concat([dfresult, dfPclass], axis=1)
    # Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfSex], axis=1)
    # Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

def creat_network():
    net = nn.Sequential()
    net.add_module("linear1", nn.Linear(in_features=15, out_features=20))
    net.add_module("relu1", nn.ReLU())
    net.add_module("linear2", nn.Linear(in_features=20,out_features= 18))
    net.add_module("relu2", nn.ReLU())
    net.add_module("linear3", nn.Linear(in_features=18, out_features=15))
    net.add_module("relu3", nn.ReLU())
    net.add_module("linear4", nn.Linear(in_features=15, out_features= 1))
    net.add_module("sigmoid", nn.Sigmoid())
    return net

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, TensorDataset

    dftrain_raw = pd.read_csv('./data/titanic/train.csv')
    dftest_raw = pd.read_csv('./data/titanic/test.csv')
    attribute =   dftrain_raw.columns
    print(   attribute)

    # ax = dftrain_raw['Survived'].value_counts().plot(kind='bar',
    #                                                  figsize=(12, 8), fontsize=15, rot=0)
    # ax.set_ylabel('Counts', fontsize=15)
    # ax.set_xlabel('Survived', fontsize=15)
    # plt.show()
    #
    # ax = dftrain_raw['Age'].plot(kind='hist', bins=20, color='purple',
    #                              figsize=(12, 8), fontsize=15)
    #
    # ax.set_ylabel('Frequency', fontsize=15)
    # ax.set_xlabel('Age', fontsize=15)
    # plt.show()
    # ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density',
    #                                                     figsize=(12, 8), fontsize=15)
    # dftrain_raw.query('Survived == 1')['Age'].plot(kind='density',
    #                                                figsize=(12, 8), fontsize=15)
    # ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
    # ax.set_ylabel('Density', fontsize=15)
    # ax.set_xlabel('Age', fontsize=15)
    # plt.show()
    x_train = processData(dftrain_raw).values
    y_train  = dftrain_raw[['Survived']].values

    x_test = processData(dftest_raw).values
    y_test = dftest_raw[['Survived']].values

    print("x_train.shape =", x_train.shape)
    print("x_test.shape =", x_test.shape)

    print("y_train.shape =", y_train.shape)
    print("y_test.shape =", y_test.shape)
    # xtrain is features ,ytrain is labels
    dl_train = DataLoader(TensorDataset(torch.tensor(x_train).float(),
                                        torch.tensor(y_train).float()
                                        ),
                     shuffle = True, batch_size = 8
                          )
    dl_valid  = DataLoader(TensorDataset(torch.tensor(x_test).float(),
                                         torch.tensor(y_test).float()),
                     shuffle = False, batch_size = 8)
    # for features , labels in dl_train:
    #     print(features,labels)
    #     # only print one example
    #     break

    net = creat_network()
    print('net is  ' , net)
    # from torchkeras import summary
    #
    # print(' data is ',summary(net, input_shape=(15,)) ) # >1.3.0 版本可以显示
    # # 开始训练
    # 通用的脚本形式
    from sklearn.metrics import accuracy_score

    loss_func = nn.BCELoss() #   bceloss
    optimizer = torch.optim.Adam(params=net.parameters(),lr = 0.01)
    metric_func = lambda y_pred,y_true: accuracy_score(y_true.data.numpy(),y_pred.data.numpy()>0.5)
    metric_name = "accuracy"

    for train in range(1):
        # 开始训练
        epochs_times = 10
        log_step_freq = 30

        dfhistory = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
        print("Start Training...")
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("==========" * 8 + "%s" % nowtime)

        for epoch in range(1, epochs_times + 1):

            # 1，训练循环-------------------------------------------------
            net.train()
            loss_sum = 0.0
            metric_sum = 0.0
            step = 1

            for step, (features, labels) in enumerate(dl_train, 1):

                # 梯度清零
                optimizer.zero_grad()

                # 正向传播求损失
                predictions = net(features)
                loss = loss_func(predictions, labels)
                metric = metric_func(predictions, labels)

                # 反向传播求梯度
                loss.backward()
                optimizer.step()

                # 打印batch级别日志
                loss_sum += loss.item()
                metric_sum += metric.item()
                if step % log_step_freq == 0:
                    print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                          (step, loss_sum / step, metric_sum / step))

            # 2，验证循环-------------------------------------------------
            net.eval()
            val_loss_sum = 0.0
            val_metric_sum = 0.0
            val_step = 1

            for val_step, (features, labels) in enumerate(dl_valid, 1):
                # 关闭梯度计算
                with torch.no_grad():
                    predictions = net(features)
                    val_loss = loss_func(predictions, labels)
                    val_metric = metric_func(predictions, labels)
                val_loss_sum += val_loss.item()
                val_metric_sum += val_metric.item()

            # 3，记录日志-------------------------------------------------
            info = (epoch, loss_sum / step, metric_sum / step,
                    val_loss_sum / val_step, val_metric_sum / val_step)
            dfhistory.loc[epoch - 1] = info # 保存日记信息

            # 打印epoch级别日志
            print(("\nEPOCH = %d, loss = %.3f," + metric_name + \
                   "  = %.3f, val_loss = %.3f, " + "val_" + metric_name + " = %.3f")
                  % info)
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("\n" + "==========" * 8 + "%s" % nowtime)

        print('Finished Training...')
    print(dfhistory)
    plot_metric(dfhistory , 'loss')
    plot_metric(dfhistory, "accuracy")

    # 使用模型
    y_pred_probs = net(torch.tensor(x_test[0:10]).float()).data
    y_pred_probs

    # 预测类别
    y_pred = torch.where(y_pred_probs > 0.5,
                         torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    y_pred

    # 保存模型
    print(net.state_dict().keys())

    # 保存模型参数

    torch.save(net.state_dict(), "./data/net_parameter.pkl")

    net_clone = creat_network()
    net_clone.load_state_dict(torch.load("./data/net_parameter.pkl"))

    net_clone.forward(torch.tensor(x_test[0:10]).float()).data

    # 保存完整模型(不推荐)

    torch.save(net, './data/net_model.pkl')
    net_loaded = torch.load('./data/net_model.pkl')
    net_loaded(torch.tensor(x_test[0:10]).float()).data