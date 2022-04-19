from sys import maxsize
from turtle import update
from sklearn.cluster import KMeans
import torch
import numpy as np
import heapq
import math

from LossNet import LossPred
from prioritized_memory import Memory

class ExperiencePool:
    def __init__(self, n_features, n_maxexps, n_clusters, select_mode="Random"):
        self.n_features = n_features
        self.n_exps = 0
        self.n_maxexps = n_maxexps
        self.n_clusters = n_clusters
        self.select_mode = select_mode
        if select_mode != "LossPER":
            self.memory = np.zeros((self.n_maxexps, n_features))
        elif select_mode == "LossPER":
            self.memory = Memory(n_maxexps)
        if select_mode == "DisSample" or select_mode == "MaxDisSample":
            self.dis = np.zeros((self.n_maxexps)) #idx-angle
        self.memory_iter = 0
        self.e_base = 0
        self.daggerMem = np.array([[]])
        self.daggerMemNum = 0
        self.is_build = False
        self.is_lossNet = False

    # 计算中心点之间的距离，大于最大距离则要新建一类
    def maxMinDisBetMeans(self):
        self.maxMeanDis = 0
        self.minMeanDis = 1e9
        for center1 in self.cluster_mean:
            for center2 in self.cluster_mean:
                self.maxMeanDis = max(self.maxMeanDis, np.linalg.norm(center1-center2))
                if (center1 == center2).all():
                    continue
                self.minMeanDis = min(self.minMeanDis, np.linalg.norm(center1-center2))
    
    # 更新最大中心点
    def updateMaxMinMeanDis(self, center_id):
        for center in self.cluster_mean:
            self.maxMeanDis = max(self.maxMeanDis, np.linalg.norm(center-self.cluster_mean[center_id]))
            if (center == self.cluster_mean[center_id]).all():
                continue
            self.minMeanDis = min(self.minMeanDis, np.linalg.norm(center-self.cluster_mean[center_id]))

    # 点与其他中心的距离
    def dis2center(self, data):
        dis = []
        for center in self.cluster_mean:
            dis.append(np.linalg.norm(data-center))
        return dis

    # 点与自身中心点的距离
    def dis2selfcenter(self, data):
        dis = self.dis2center(data)
        return np.argmin(dis).item(), np.min(dis).item()

    def updateMean(self, data, center, where):
        cp = center
        new_center = center + where * (1 / self.n_maxexps) * (center - data)
        return new_center, np.sum(new_center - cp)

    # 拉近或疏远中心点与自身距离，拉近属于自己类的中心点，属于非自己的中心点，如果超过最大中心点距离（阈值）则自建中心点
    # updateMode 1-插入 -1-删除 
    def updateMeans(self, data, updateMode):
        center_id, center_dis = self.dis2selfcenter(data)
        for id in range(self.n_clusters):
            if center_id == id:
                self.cluster_mean[id], _ = self.updateMean(data, self.cluster_mean[id], -1 * updateMode)
                continue
            self.cluster_mean[id], _ = self.updateMean(data, self.cluster_mean[id], 1 * updateMode)

    def calculate(self, a, b):
        a = a.reshape(-1, a.shape[0])
        b = b.reshape(-1, b.shape[0])
        cos_ = np.dot(a, b.transpose())/(np.linalg.norm(a)*np.linalg.norm(b))
        arccos2_ = np.arccos(cos_)
        if math.isnan(arccos2_):
            return 0
        return arccos2_

    # 添加数据
    # 若未建立且未满则直接插入
    # 若建立了则将过时信息删去并调整中心点
    def add(self, data):
        if self.select_mode == "DisSample":
            self.memory[self.memory_iter] = data
            if self.is_build:
                self.dis[self.memory_iter] = self.calculate(data, self.e_base)
            self.memory_iter += 1
            if self.memory_iter == self.n_maxexps:
                self.is_build = True
                self.e_base = np.mean(self.memory, axis=0)
                for i in range(self.n_maxexps):
                    self.dis[i] = self.calculate(self.memory[i, :], self.e_base)
            self.memory_iter %= self.n_maxexps

        elif self.select_mode == "MaxDisSample":
            self.memory[self.memory_iter] = data
            if self.is_build:
                self.dis[self.memory_iter] = self.calculate(data, self.e_base)
            self.memory_iter += 1
            if self.memory_iter == self.n_maxexps:
                self.is_build = True
                self.e_base = np.mean(self.memory, axis=0)
                for i in range(self.n_maxexps):
                    self.dis[i] = self.calculate(self.memory[i, :], self.e_base)
            self.memory_iter %= self.n_maxexps
        
        elif self.select_mode != "LossPER":
            copy = self.memory[self.memory_iter]
            self.memory[self.memory_iter] = data
            self.memory_iter += 1
            if self.memory_iter == self.n_maxexps:
                #self.kmeans = KMeans( 
                #    n_clusters = self.n_clusters,
                #    n_init = 10,
                #    max_iter = 300,
                #    init = 'k-means++',
                #    ).fit(self.memory)
                #print("------------------K-Means建立------------------")
                #self.cluster_mean = self.kmeans.cluster_centers_
                self.is_build = True
                #self.maxMinDisBetMeans()
            #elif self.is_build:
            #    self.updateMeans(data, 1)
            #    self.updateMeans(copy, -1)
            self.memory_iter %= self.n_maxexps
        
        elif self.select_mode == "LossPER":
            if not self.is_lossNet:
                self.LossPred = LossPred(self.n_features)
                self.is_lossNet = True
            self.memory_iter += 1
            now_data = data
            self.memory.add(self.LossPred.lossNet(torch.from_numpy(data).to(torch.float32)).detach(), now_data)
            if self.memory_iter == self.n_maxexps:
                self.is_build = True
            self.memory_iter %= self.n_maxexps

    # SumTree的update
    def update_SumTree(self, batch_num, idxs, states):
        for i in range(batch_num):
            idx = idxs[i]
            error = self.LossPred.lossNet(states[i]).detach()
            self.memory.update(idx, error)

    # 挑选离k-means中心最远的点
    def maxDis2Center_Sample(self, batch_size):
        heap = [[] for i in range(self.n_clusters)]
        batch_data = []
        for data_id in range(self.n_maxexps):
            center_id, center_dis = self.dis2selfcenter(self.memory[data_id])
            heapq.heappush(heap[center_id], (-center_dis, data_id))
        i = 0
        t = 0
        while i < batch_size:
            if len(heap[t%self.n_clusters])==0:
                t += 1
                continue
            select_data = heapq.heappop(heap[t%self.n_clusters])
            batch_data.append(self.memory[select_data[1], :])
            t += 1
            i += 1
        return np.array(batch_data)

    def Random_Sample(self, batch_size):
        sample_index = np.random.choice(self.n_maxexps, batch_size)
        return self.memory[sample_index, :]

    def maxEntropy_Sample(self, batch_size, model):
        heap = []
        batch_data = []
        def calculate_Entropy(model, data):
            res = 0
            data = torch.FloatTensor(data).detach()
            output = model(data).detach()
            for x in output:
                res += x * torch.log(x)
            return res.item()
        for data_id in range(self.n_maxexps):
            entropy = calculate_Entropy(model, self.memory[data_id])
            heapq.heappush(heap, (entropy, data_id))
        i = 0
        while i < batch_size:
            select_data = heapq.heappop(heap)
            batch_data.append(self.memory[select_data[1], :])
            i += 1
        return np.array(batch_data)

    # 需要多个网络的QBC模式
    def QueryByCommittee(self, batch_size, model):
        pass

    def DensityWeighted(self, batch_size, model, beta):
        heap = []
        batch_data = []
        def calculate_Entropy(model, data):
            res = 0
            data = torch.FloatTensor(data).detach()
            output = model(data).detach()
            for x in output:
                res += x * torch.log(x)
            return res.item()
        def calculate_similarity(data):
            res = 0
            for center in self.cluster_mean:
                res += np.sum(np.abs(data - center))
            return res.item() / self.n_clusters
        for data_id in range(self.n_maxexps):
            entropy = calculate_Entropy(model, self.memory[data_id])
            similarity = pow(calculate_similarity(self.memory[data_id]), beta)
            weight = entropy * similarity
            heapq.heappush(heap, (weight, data_id))
        i = 0
        while i < batch_size:
            select_data = heapq.heappop(heap)
            batch_data.append(self.memory[select_data[1], :])
            i += 1
        return np.array(batch_data)

    def LossWeighted(self, batch_size):
        heap = []
        batch_data = []
        if not self.is_lossNet:
            self.LossPred = LossPred(self.n_features)
            self.is_build = True
        for data_id in range(self.n_maxexps):
            loss = self.LossPred.pred(torch.FloatTensor(self.memory[data_id]))
            heapq.heappush(heap, (-loss.item(), data_id))
        i = 0
        t = 0
        while i < batch_size:
            select_data = heapq.heappop(heap)
            batch_data.append(self.memory[select_data[1], :])
            t += 1
            i += 1
        return np.array(batch_data)

    def LossPER(self, batch_size):
        return self.memory.sample(batch_size)

    def LossPredTrain(self, data, yhat_loss):
        mean_loss = self.LossPred.train(data, yhat_loss)
        return mean_loss

    def DisSample(self, batch_size):
        minn = np.min(self.dis)
        gap = (np.max(self.dis) - np.min(self.dis))/batch_size
        batch_data = []
        batch_num = batch_size+2
        heap = [[] for i in range(batch_num)]
        for i in range(self.n_maxexps):
            dist = self.dis[i] - minn
            index = int(dist // gap)
            heapq.heappush(heap[index], (-dist, i))
        i = 0
        t = 0
        while i < batch_size:
            if len(heap[t%batch_num])==0:
                t += 1
                continue
            select_data = heapq.heappop(heap[t%batch_num])
            batch_data.append(self.memory[select_data[1], :])
            t += 1
            i += 1
        return np.array(batch_data)

    def MaxDisSample(self, batch_size):
        heap = []
        batch_data = []
        for i in range(self.n_maxexps):
            heapq.heappush(heap, (-abs(self.dis[i]), i))
        i = 0
        t = 0
        while i < batch_size:
            select_data = heapq.heappop(heap)
            batch_data.append(self.memory[select_data[1], :])
            t += 1
            i += 1
        return np.array(batch_data)

    
    # 挑选样本
    def sample(self, batch_size, model, beta=0.9):
        if self.select_mode == "maxDis2Center":
            return self.maxDis2Center_Sample(batch_size)
        
        elif self.select_mode == "Random":
            return self.Random_Sample(batch_size)
        
        elif self.select_mode == "MaxEntropy":
            return self.maxEntropy_Sample(batch_size, model)

        elif self.select_mode == "QueryByCommittee":
            return self.QueryByCommittee(batch_size, model)

        elif self.select_mode == "Density-Weighted":
            return self.DensityWeighted(batch_size, model, beta)
        
        elif self.select_mode == "LossPredict":
            return self.LossWeighted(batch_size)

        elif self.select_mode == "LossPER":
            return self.LossPER(batch_size)
        
        elif self.select_mode == "DisSample":
            return self.DisSample(batch_size)

        elif self.select_mode == "MaxDisSample":
            batch_data = self.MaxDisSample(batch_size)
            self.toDaggerMem(batch_data)
            return batch_data

    def npNorm(self, data):
        return data / np.linalg.norm(data)

    def toDaggerMem(self, batch_data):
        if self.daggerMemNum < 32:
            self.daggerMemNum = batch_data.shape[0]
            for data in batch_data:
                data = self.npNorm(data)
                print(np.linalg.norm(data))
            self.e_base = np.mean(batch_data, axis=0)
            self.e_base = self.npNorm(self.e_base)
        else:
            for data in batch_data:
                data = self.npNorm(data)
            self.e_base = self.e_base*self.daggerMemNum + np.sum(batch_data, axis=0)
            self.daggerMemNum += batch_data.shape[0]
            self.e_base /= self.daggerMemNum
            self.e_base = self.npNorm(self.e_base)
        for i in range(self.n_maxexps):
            self.dis[i] = self.calculate(self.memory[i, :], self.e_base)
        

    #def sample(self, batch_size):
    #    sample_index = np.random.choice(self.daggerMem.shape[0], batch_size)
    #    return self.daggerMem[sample_index, :]
        
 