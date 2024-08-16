# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from fl_library.utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        # 设置PyTorch的随机种子，随机种子决定了生成器的初始状态，使用相同的随机种子就能保证产生相同随机数序列，以确保实验的可重复性，
        torch.manual_seed(0)
        # 将args.model的副本给self.model
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        # 设置训练是否慢速
        self.train_slow = kwargs['train_slow']
        # 设置发送是否慢速
        self.send_slow = kwargs['send_slow']
        # 初始化一个字典，用于记录训练轮次，训练时间
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        # 初始化一个字典，用于记录发送轮次，发送时间
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # 设置模型隐私保护
        self.privacy = args.privacy
        # 设置差分隐私中的噪声标准差
        self.dp_sigma = args.dp_sigma

        # 定义损失函数为交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        # 定义优化器为随机梯度下降SGD，并传入模型参数与学习率
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 定义学习率调度器为指数衰减调度器（ExponentialLR），并传入优化器和衰减因子
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        # 设置学习率的衰减因子
        self.learning_rate_decay = args.learning_rate_decay

    # 加载训练集
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # 用read_client_data函数读取数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    # 加载测试集，与训练集相同
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    # 将模型传入
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    # 模型克隆，将一个模型复制到另一个模型
    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    # 更新模型参数
    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    # 用于评估模型性能的方法
    def test_metrics(self):
        # 加载测试集
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # 将模型设置为评估模式
        self.model.eval()

        # 记录测试准确度
        test_acc = 0
        # 记录测试样本数目
        test_num = 0
        # 用于记录预测标签
        y_prob = []
        # 用于存储真实标签
        y_true = []
        
        with torch.no_grad():
            # 获取测试集的x和y
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    # 用于计算模型在训练集上的性能指标，损失值
    def train_metrics(self):
        # 加载训练集
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # 将模型设为评估模式，关闭了训练时使用的 Dropout 和 Batch Normalization 层的功能，并将模型置于不计算梯度的模式
        self.model.eval()

        # 训练样本总数
        train_num = 0
        # 记录累积损失值
        losses = 0
        # 关闭梯度计算上下文，进行训练数据推断
        with torch.no_grad():
            # 循环获取数据集的输入x，y
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 将数据输入模型，得到输出
                output = self.model(x)
                # 累积计算损失值
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    # 用于保存模型参数
    # item为要保存的项
    # item_name为要保存的名称
    # item_path为要保存的路径
    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # 用于加载模型
    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
