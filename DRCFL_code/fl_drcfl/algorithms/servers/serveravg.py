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

import time
from fl_library.algorithms.clients.clientavg import clientAVG
from fl_library.algorithms.servers.serverbase import Server
from threading import Thread


class FedAvg(Server):
    def __init__(self, args, times):
        # 调用父类初始化方法
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        # 调用服务器方法，创建ClientAVG客户端
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        # 迭代次数为全局更新次数
        for i in range(self.global_rounds+1):
            # 记录当前轮次开始时间
            s_t = time.time()
            # 调用父类中选择客户端的函数
            self.selected_clients = self.select_clients()
            # 发送模型给客户端
            self.send_models()

            # 如果当前轮次可以进行评估全局模型
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                # 调用评估全局模型性能方法
                self.evaluate()

            # 对于每个被选定的客户端执行以下操作
            for client in self.selected_clients:
                # 训练该客户端
                client.train()

                # print(client.id)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            # 接收从客户端得到的模型
            self.receive_models()
            # 如果开启了DLG评估
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            # 聚合来自客户端的模型参数
            self.aggregate_parameters()

            # 记录当前轮次开销，并记录在budget中
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # 如果开启了自动终止功能，达成条件就可以终止
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 最后结果输出打印
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # 最佳精度
        print(max(self.rs_test_acc))
        # 计算平均每轮时间消耗
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # 保存训练结果
        self.save_results()
        # 保存全局模型
        self.save_global_model()

        # 如果存在新客户端
        if self.num_new_clients > 0:
            # 设置评估新客户端标志为True
            self.eval_new_clients = True
            # 为新客户端设置参数
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
