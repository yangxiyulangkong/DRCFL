import copy
import time
import torch
import torch.nn as nn
from fl_drcfl.algorithms.clients.client_DRCFL import client_DRCFL
from fl_drcfl.algorithms.servers.serverbase import Server
from torch.utils.data import DataLoader

class DRCFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None
        self.set_slow_clients()
        self.set_clients(client_DRCFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.euclidean_distance_threshold = args.euclidean_distance_threshold
        self.head = self.clients[0].model.head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)

    def train(self):
        s_t = time.time()
        self.send_models()

        for i in range(self.global_rounds+1):
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            if ((i == self.round_add_new_clients) & (self.num_new_clients > 0)):
                self.newhead = self.clients[0].model.head
                self.set_new_clients(client_DRCFL)
                self.new_client_send_models()
                self.clients.extend(self.new_clients)
                self.num_clients = self.num_clients + self.num_new_clients

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            self.receive_protos()
            self.clustering()
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            s_t = time.time()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        self.save_results()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.head)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def new_client_send_models(self):
        assert (len(self.new_clients) > 0)

        for client in self.new_clients:
            start_time = time.time()
            client.set_parameters(self.newhead)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.id, client.protos[cc], y))
    def clustering(self):
        c_id_list = []
        p_list = []
        y_list = []
        self.sample_clusters = []
        self.clu_max = 0

        proto_loader = DataLoader(self.uploaded_protos, drop_last=False, shuffle=True)

        for c_id, p, y in proto_loader:
            c_id_list.append(c_id)
            p_list.append(p)
            y_list.append(y)

        num_points = len(p_list)
        clusters = [-1] * num_points
        cluster_index = 0
        threshold = self.euclidean_distance_threshold

        for i in range(num_points):
            if clusters[i] == -1:
                clusters[i] = cluster_index
                ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                self.sample_clusters.append((c_id_list[i], y_list[i], ccluster_index, p_list[i]))

                for j in range(i + 1, num_points):
                    distance = torch.sqrt(torch.sum((p_list[i] - p_list[j]) ** 2))
                    if distance < threshold:
                        if clusters[j] == -1:
                            clusters[j] = cluster_index
                            ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                            self.sample_clusters.append((c_id_list[j], y_list[j], ccluster_index, p_list[j]))
                cluster_index += 1
        self.clu_max = cluster_index

    def train_head(self):
        id_list = []
        complete_head_list = []
        head_list = []

        self.head_copy = self.head
        proto_loader = self.sample_clusters

        for i in range(self.clu_max):
            self.head = self.head_copy
            for c_id, y, clu_id, p in proto_loader:
                if clu_id == i:
                    out = self.head(p)
                    loss = self.CEloss(out, y)
                    self.opt_h.zero_grad()
                    loss.backward()
                    self.opt_h.step()

            complete_head_list.append(copy.deepcopy(self.head))

        num_clients = len(self.selected_clients)
        num_classes = len(proto_loader) / num_clients
        num_classes = int(num_classes)

        i = 0
        total_head = 0
        sorted_proto_loader = sorted(proto_loader, key=lambda x: x[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for c_id, y, clu_id, p in sorted_proto_loader:
            i = i + 1
            total_head = total_head + complete_head_list[clu_id].weight.data
            if(i % num_classes == 0):
                avg_head = total_head/num_classes
                new_linear = nn.Linear(in_features=512, out_features=100, bias=True)
                new_linear.weight.data = nn.Parameter(avg_head)
                new_linear.to(device)
                head_list.append(new_linear)
                id_list.append(c_id)
                i = 0
                total_head = 0

        for client in self.selected_clients:
            start_time = time.time()
            k = 0
            for id in id_list:
                if client.id == id:
                    client.set_parameters(head_list[k])
                k = k+1

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

        total_head = 0
        for head_cid in head_list:
            total_head = total_head + head_cid.weight.data

        avg_head = total_head/num_clients
        new_linear = nn.Linear(in_features=512, out_features=100, bias=True)
        new_linear.weight.data = nn.Parameter(avg_head)
        new_linear.to(device)
        self.head = new_linear