# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import sys

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

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from fl_library.utils.data_utils import read_client_data
from fl_library.utils.dlg import DLG

# Server类
class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client
        self.round_add_new_clients = args.round_add_new_clients
        self.num_new_clients = args.num_new_clients

        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    # 用于服务端设置客户端内容（创建客户端）
    def set_clients(self, clientObj):
        # 循环，i表示客户端数量，将客户端索引，训练速度，发送速度打包成元组
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # 读取第i个客户端的训练数据
            train_data = read_client_data(self.dataset, i, is_train=True)
            # 读取第i个客户端的测试数据
            test_data = read_client_data(self.dataset, i, is_train=False)
            # 创建一个客户端对象，使用ClientObj实例化对象
            client = clientObj(self.args,
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            # 将创建的客户端添加到clients中
            self.clients.append(client)

    # random select slow clients
    # 随机选择慢训练客户端
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    # 设置慢训练客户端
    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    # 选择客户端的函数
    def select_clients(self):
        # 如果为True，则随机选择一部分客户端
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            # print(self.num_join_clients)
            # print(self.join_ratio)
            self.current_num_join_clients = self.num_join_clients
        # 从所有客户端中随机选择当前轮次参与联邦学习的客户端
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    # 实现服务器向客户端发送全局模型功能
    def send_models(self):
        # 用断言确保客户端列表至少有一个客户端对象
        assert (len(self.clients) > 0)

        # 给所有客户端发送模型
        for client in self.clients:
            # 记录开始发送时间
            start_time = time.time()

            # 将全局模型参数发送给客户端
            client.set_parameters(self.global_model)

            # 向客户端发送时间
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # 接收客户端的模型
    def receive_models(self):
        # 断言，确保知识有一个客户端被选中
        assert (len(self.selected_clients) > 0)

        # 从被选中的客户端中随机选择一部分作为客户端，参数比例由client_drop_rate控制，若为0，则表示所有均为活跃客户端
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        # 存储客户端id
        self.uploaded_ids = []
        # 存储客户端训练样本数量
        self.uploaded_weights = []
        # 存储客户端模型
        self.uploaded_models = []
        tot_samples = 0
        # 变量所有活跃客户端
        for client in active_clients:
            # 计算客户端平均时间消耗，包括训练时间，发送时间
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            # 检查客户端的时间消耗是否小于等于预设的时间阈值time_threthold
            if client_time_cost <= self.time_threthold:
                # 累加总的训练样本数量
                tot_samples += client.train_samples
                # 将满足条件的id添加到列表
                self.uploaded_ids.append(client.id)
                # 每个客户端段的权重可以先视作该客户端参与训练的样本总数
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)

        # 对上传的权重进行归一化出来，每个客户端参与训练的数量除以总的训练样本数量，确保权重总和为1
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # 创建一个全局模型将其初始化为列表中第一个模型进行深度拷贝
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # 循环遍历全局模型所有参数，并将其置零
        for param in self.global_model.parameters():
            param.data.zero_()

        # 循环遍历上传的客户端更新的模型权重，然后调用add_parameters进行加权求和
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        # 循环迭代全局模型参数和客户端模型参数，并逐个配对
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            # 每次迭代对全局模型参数进行更新，具体是将各个client_param的参数数据乘以其权重w，再累加
            server_param.data += client_param.data.clone() * w

    # 用于保存全局模型
    def save_global_model(self):
        # 构建保存模型的路径，文件夹名为model
        model_path = os.path.join("models", self.dataset)
        # 如果路径不存在，则创建
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 文件名称为算法名+_server.pt
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        # 保存全局模型和路径
        torch.save(self.global_model, model_path)

    # 用于加载全局模型
    def load_model(self):
        # 构建了保存模型的文件夹路径，其中models是存放模型的文件夹名称，self.dataset是数据集名称
        model_path = os.path.join("models", self.dataset)
        # 拼接路径
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        # 断言判断模型文件是否存在
        assert (os.path.exists(model_path))
        # 使用torch.load加载保存的全局模型
        self.global_model = torch.load(model_path)

    # 判断模型是否存在
    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    # 存储结果代码
    def save_results(self):
        # 生成文件名
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        # 如果不存在路径穿件
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # 检查self.rs_test_acc是否包含数据
        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            # 使用h5py创建一个HDF5格式的文件对象
            with h5py.File(file_path, 'w') as hf:
                # 写入结果集
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    # 保存项目
    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    # 加载项目
    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    # 测试集评价指标
    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    # 训练集评价指标函数
    def train_metrics(self):
        # 检查是否有需要评估的客户端
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    # 评估所选客户端的性能
    def evaluate(self, acc=None, loss=None):
        # 获取测试集指标
        stats = self.test_metrics()
        # 获取训练集指标
        stats_train = self.train_metrics()

        # 平均准确率=所有客户端的准确率之和/该轮参与客户端数
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        # 计算平均AUC
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # 计算平均训练损失，所有客户端损失相加/客户端数量
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    # 设置新客户端
    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)


    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
