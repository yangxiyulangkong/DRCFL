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
import numpy as np
import time
from fl_library.algorithms.clients.clientbase import Client
from fl_library.utils.privacy import *

# 定义了一个clientAVG的子类，继承Client
class clientAVG(Client):
    # 定义了子类初始化的方法，接收一些参数
    # args包含各种设置的参数对象
    # id客户端唯一标识符
    # train_samples用于训练的样本数据
    # test_samples用于测试的样本数据
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    # 定义了训练方法train()，用于在客户端上执行模型训练过程
    def train(self):
        # 调用父类中load_train_data()方法，加载训练集
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        # 将模型设置为训练模式
        self.model.train()

        # differential privacy
        # 隐私保护设置，如果为真，调用差分等方法进行初始化
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        # 如果设置了train.slow，采用随机减少本地训练次数的方式
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 循环次数为每一轮联邦学习，本地需要更新迭代的次数
        for epoch in range(max_local_epochs):
            # 循环遍历加载的数据，i表示索引，（x，y）表示当前批次的特征和标签
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                # 将y标签也送到设备中
                y = y.to(self.device)
                # 如果启用了慢速训练
                if self.train_slow:
                    # 则每个批次训练之后，随机等待一段时间
                    time.sleep(0.1 * np.abs(np.random.rand()))
                # 将输入x送到模型中前向传播，得到输出output
                output = self.model(x)
                # 计算模型输出和实际标签y之间的损失值
                loss = self.loss(output, y)
                # 将优化器中之前的梯度置零
                self.optimizer.zero_grad()
                # 执行反向传播，计算损失函数对模型参数的梯度
                loss.backward()
                # 根据计算得到的梯度更新模型参数，执行一步优化器的参数更新
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
