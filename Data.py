# -*- coding: utf-8 -*-
import pickle
from collections import Counter

import torch
from torch import softmax
from torch.utils.data import Dataset  # 导入DataSet类
import numpy as np  # 导入numpy包

# 在__init__中，train_path后面, item_length_path
class DataSetTrain(Dataset):

    def __init__(self, train_path,user_count=0, item_count=0, neg_size=5, hist_len=2,
                 directed=False):  # 定义类属性：训练集路径，用户数量，music数量，负样本数，his_len,directed
        self.neg_size = neg_size
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.directed = directed

        self.NEG_SAMPLING_POWER = 0.75  # 定义对象属性，不需要传入。不知道干啥用的。
        self.neg_table_size = int(1e8)  # 定义对象属性，不需要传入。不知道干啥用的。1e8是1*10的8次方。1e-8是-8次方

        self.node2hist = dict()  # 定义一个空的字典，node2hist
        self.user_item_dict = dict()  # 定义一个空的字典，user_item_dict
        self.user_node_set = set()  # 定义一个空的集合，uesr_node_set
        self.item_node_set = set()  # 定义一个空的集合，item_node_set
        self.degrees = dict()  # 定义一个空的字典，degrees

        self.user_item_dict_list = dict()

        self.cat_node_set = set()  # 定义一个空的集合，用于存储poi所属类别的序号

        with open(train_path, 'r') as infile:  #以只读的方式加载训练集路径，命名为infile
            for line in infile:  # 对训练集进行遍历
                parts = line.strip().split()  # strip函数去除字符串开头与结尾的字符，默认为空格。split函数用于分割字符串，默认按照空字符进行分割。
                s_node = int(parts[0])  # 将parts[0]（在index2item中对应的用户序号）转换为int类型
                t_node = int(parts[1])  # 用户的倾听记录序号。（在index2item中对应的序号）转换成int类型
                time_stamp = float(parts[2])  # 将parts[2]转换成float类型。距离第一首歌的时间

                if s_node not in self.user_item_dict:  # 如果用户序号不在这个构建的字典中，
                    self.user_item_dict[s_node] = set()  # 将这个字典中的用户序号对应的value值构建为一个空的集合


                    self.user_item_dict_list[s_node] = list()  # 这一步是为了得到每一个用户的poi访问记录，用列表存储，重复的不删除
                self.user_item_dict[s_node].add(t_node)  # 将用户倾听记录序号添加到用户序号对应的集合中

                self.user_item_dict_list[s_node].append(t_node)  # 将记录添加进来，为了get_item使用

                self.user_node_set.add(s_node)  # 将parts[0]（用户id）添加到user_node_set集合中
                self.item_node_set.add(t_node)  # 将parts[1](用户倾听记录)添加到item_node_set集合中

                if s_node not in self.node2hist:  # 如果用户序号不在node2hist字典中，
                    self.node2hist[s_node] = list()  # 就将node2hist字典中的用户序号对应的value值构建为一个空的列表
               # self.node2hist[s_node].append((t_node, time_stamp, interval_time, length_time))  # 将这个用户序号对应的音乐序号，时间戳，时间差，时间长度添加到这个列表中
                self.node2hist[s_node].append((t_node, time_stamp))

                if not directed:  # 如果directed为FALSE，这个结果为TRUE，执行下列代码。如果directed为TRUE，这个结果为FALSE，不执行下列代码。。。在Model中，directed为TRUE，所以，不执行下列代码。
                    if t_node not in self.node2hist:  # 如果music序号不在node2hist中的话，执行下列代码
                        self.node2hist[t_node] = list()  # 将node2hist字典中music序号对应的value值构建为一个空的列表
                   # self.node2hist[t_node].append((s_node, time_stamp, interval_time, length_time))  # 然后将列表中添加用户序号，时间戳，时间差，时间长度
                    self.node2hist[t_node].append((s_node, time_stamp))


                if s_node not in self.degrees:  # 如果用户序号不在degrees字典中，执行下列代码
                    self.degrees[s_node] = 0  # 将degrees字典中的用户序号对应的value值设为0
                if t_node not in self.degrees:  # 如果music序号不在degrees字典中的话，执行下列代码
                    self.degrees[t_node] = 0  # 将degrees中的music序号对应的value值设为0
                self.degrees[s_node] += 1  # 用户序号对应的value值加1
                self.degrees[t_node] += 1  # music序号对应的value值加1


        self.node_dim = self.user_count + self.item_count

        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        self.neg_table = np.zeros((self.neg_table_size,))  # 返回一个neg_table_size大小的0向量。  neg_table_size大小为1e8.10000000
        self.init_neg_table()  # 调用init_neg_table方法

    def get_node_dim(self):  # get_node_dim函数，用于返回用户数量与音乐数量的合计
        return self.node_dim

    def init_neg_table(self):  # 用于初始化负样本表
        total_sum, cur_sum, por = 0., 0., 0.  # 初始化为0
        n_id = 0  # 将n_id初始化为0
        for k in range(self.node_dim):  # node_dim是用户总数加上music总数。从1遍历到node_dim
            if k in self.degrees:  # 如果这个数在degree字典中的话，即：k=某一个key值，执行下列代码
                total_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)  # power为x的几次方。将total_num等于k对应的value值的NEG_SAMPLING_POWER次方，为0.75。对total_num进行一个累加
            else:
                self.degrees[k] = 0  # 如果不在degree字典中的话，就将字典中k对应的value值设置为0。本来degrees字典中，只有训练集中的数据。经过这个for循环之后得到的degrees字典，是所有的用户加起来以及音乐加起来之后的字典。不在训练集中的数据以及被删除的数据对应的value会设置为0
        for k in range(self.neg_table_size):  # for循环，从1到neg_table_size，进行这么多轮循环。从1到1e8.一亿
            if (k + 1.) / self.neg_table_size > por:  # 如果（k+1）/neg_table_size>por,执行下列代码
                while self.degrees[n_id] == 0:  # 当degree[n_id]==0的时候，进行下列循环
                    n_id += 1  # 计算degrees字典中有多少个0值，即不存在训练集中值以及被删除的值

                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)  # 将cur_sum = degrees字典中，n_id对应的value值的NEG_SAMPLING_POWER次方。
                por = cur_sum / total_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def __len__(self):  # 返回所有用户的倾听次数，以及所有音乐的被倾听次数
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]  # idx2source_id存放的是用户的序号。一个用户听了多少次就有多少个他的序号
        t_idx = self.idx2target_id[idx]  # idx2target_id存放的是这个用户听的记录的顺序号
        t_node = self.node2hist[s_node][t_idx][0]  # 得到字典中s_node（用户序号）对应的t_idx对应的第一个元素（不可能为用户序号，只能为音乐序号）。因为node2hist是一个key为用户id，value为这个用户听过的音乐记录信息。因为这是用户序号对应的value值
        t_time = self.node2hist[s_node][t_idx][1]  # 得到字典中指定的记录距离第一首音乐的时间

        # NYC数据集，去除poi少于5次，用户少于5次
        path1 = open('../poi_cat_dict_path', 'rb')

        poi_cat = pickle.load(path1)

        # 用于得到目标poi的所属的类别序号
        t_node_cat = poi_cat[t_node]


        user_poi_data = self.user_item_dict_list[s_node]
        path = open('../data_index_poi_name','rb')

        item2index = pickle.load(path)
        t_node_name = []

        for poi_id in user_poi_data:
            t_node_name.append(item2index[poi_id])  # 可以直接获得poi_id对应的poi名字
        count = Counter(t_node_name)  # 得到每一个poi的访问次数

        poi_weight = dict()
        for i in count:
            poi_ava = len(user_poi_data)/len(count)  # 先计算出用户对poi的平均访问次数
            poi_frequent_weight = count[i]/poi_ava  # 计算每一个poi访问次数所占的权重
            poi_weight[i] = poi_frequent_weight  # key为用户的访问记录的poi名字，value为poi访问频率所占平均访问频率的权重

        if t_idx - self.hist_len < 0:
            hist = self.node2hist[s_node][0:t_idx]  # 将hist等于node2list字典中s_node对应的value列表中从0到t_idx的值
        else:  # 如果t_idx - hist_len >= 0 的话，执行下列代码
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]  # 将hist等于node2list字典中s_node对应的value列表中 t_idx-hist_len到t_idx的值。就是要t_idx附近hist_len的值


        hist_poi_cat = []
        for i in hist:
            poi_id = i[0]
            poi_cat_id = poi_cat[poi_id]
            hist_poi_cat.append(poi_cat_id)


        hist_poi_name = []
        for i in hist:  # 获得历史序列的poiid
            hist_poi_name.append(item2index[i[0]])  # 将poi_id对应的名字添加到hist_poi_name列表中

        # 这里没有将权重设置为1，就是正常的访问频率除以平均频率。
        hist_poi_weight = []
        for i in hist_poi_name:
            hist_poi_weight.append(poi_weight[i])

        # 这里将大于1的poi频率权重设置为1.
        for i in range(len(hist_poi_weight)):
            if hist_poi_weight[i] >1:
                hist_poi_weight[i] = 1

        hist_poi_weight = torch.Tensor(hist_poi_weight)
        hist_nodes = [h[0] for h in hist]  # 对hist遍历，只要h[0]值，返回一个列表形式。即：序号值
        hist_times = [h[1] for h in hist]  # 只要hist的h[1]值，即：与第一首音乐的时间差

        # 用于获得历史访问poi的所属的类别序号
        np_h_cat = np.zeros((self.hist_len,))
        np_h_cat[:len(hist_nodes)] = hist_poi_cat

        np_h_nodes = np.zeros((self.hist_len,))  # 返回一个长度为hist_len的0向量
        np_h_nodes[:len(hist_nodes)] = hist_nodes  # 将这个向量0到len(hist_nodes)的值等于hist_nodes的值
        np_h_times = np.zeros((self.hist_len,))  # 同上
        np_h_times[:len(hist_nodes)] = hist_times

        np_h_masks = np.zeros((self.hist_len,))  # 返回一个长度为hist_len的0向量
        np_h_masks[:len(hist_nodes)] = 1.  # 从0到len(hist_nodes)，赋值为1

        np_h_weight = np.zeros((self.hist_len,))
        np_h_weight[:len(hist_nodes)] = hist_poi_weight

        neg_nodes = self.negative_sampling(s_node, t_node)  # 将s_node（用户序号），t_node（音乐序号）传入negative_sampling函数，

# 用于得到负样本的所属的类别
        neg_nodes_cat = []
        for i in neg_nodes:

            neg_poi_cat_id = poi_cat[i]
            neg_nodes_cat.append(neg_poi_cat_id)

        np_neg_nodes_cat = np.zeros((self.neg_size,))
        np_neg_nodes_cat[:len(neg_nodes)] = neg_nodes_cat

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_cat':t_node_cat,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,
            'history_masks': np_h_masks,
            'history_cat': np_h_cat,
            'neg_nodes': neg_nodes,
            'neg_node_cat':np_neg_nodes_cat,
            'history_frequent_weight':np_h_weight  # 返回历史记录中每一个poi占的频率的权重
        }  # sample

        return sample

# 生成负样本，用于训练模型
    def negative_sampling(self, source_node, target_node):  # sourec_node是某一个用户的序号值，target_node是这个序号值听的歌曲记录
        sampled_nodes = []  # 创建一个空的列表
        func = lambda: self.neg_table[np.random.randint(0, self.neg_table_size)]  # func函数为neg_table列表中的一个随机index（index大于0小于neg_table_size）的值.0---10000000
        for i in range(self.neg_size):  # 从1到neg_size进行遍历。这里neg_size为5、就是循环五次。选择5个负样本
            temp_neg_node = func()  # temp_neg_node为func返回值。
            # user_item_dict是一个字典。key为用户序号，value为这个用户听过的音乐序号。
            while temp_neg_node in self.user_item_dict[source_node] or temp_neg_node == source_node or temp_neg_node == target_node or temp_neg_node >= self.item_count:
                temp_neg_node = func()  # temp_neg_node不能为这些其中的任何一个，如果等于的话，再随机选择一个
            sampled_nodes.append(temp_neg_node)  # 将不等于这些数的时候，将其添加到sample_nodes列表中
        return np.array(sampled_nodes)  # 返回一个数组类型的sample_nodes。就是将列表转换成数组形式。就是返回负样本


class DataSetTestNext(Dataset):

    def __init__(self, file_path, data_tr, user_count=0, item_count=0, hist_len=2, directed=False):
        self.user_count = user_count
        self.item_count = item_count
        self.hist_len = hist_len
        self.directed = directed
        self.node2hist = dict()
        self.user_item_dict_list = dict()

        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.split()
                s_node = int(parts[0])
                t_node = int(parts[1])
                time_stamp = float(parts[2])
                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()

                    self.user_item_dict_list[s_node] = list()  # 这一步是为了得到每一个用户的poi访问记录，用列表存储，重复的不删除
                self.node2hist[s_node].append((t_node, time_stamp))

                self.user_item_dict_list[s_node].append(t_node)  # 将记录添加进来，为了get_item使用

                if not directed:
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    #self.node2hist[t_node].append((s_node, time_stamp, interval_time, length_time))
                    self.node2hist[t_node].append((s_node, time_stamp))

# 最终node2hist 字典类型。key1存放的是遍历后的用户以及这个用户的倾听信息。key2音乐序列：这首音乐被听的用户的序号以及信息，key3用户id：这个用户的倾听信息、、、、、、、、、
        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])
# data_size等于这个所有用户的倾听记录次数，加上所有音乐的被倾听次数

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)  # 返回一个data_size大小的向量
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)  # 返回一个data_size大小的向量，用于存储idx2target_id
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node  # 与训练集相同
                self.idx2target_id[idx] = t_idx
                idx += 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        t_time = self.node2hist[s_node][t_idx][1]


        path1 = open('../poi_cat_dict_path', 'rb')
        poi_cat = pickle.load(path1)
        t_node_cat = poi_cat[t_node]

        user_poi_data = self.user_item_dict_list[s_node]
        path = open('../data_index_poi_name','rb')


        item2index = pickle.load(path)
        t_node_name = []

        for poi_id in user_poi_data:
            t_node_name.append(item2index[poi_id])

        count = Counter(t_node_name)

        poi_weight = dict()
        for i in count:
            poi_ava = len(user_poi_data)/len(count)
            poi_frequent_weight = count[i]/poi_ava
            poi_weight[i] = poi_frequent_weight  #


        if t_idx - self.hist_len < 0:
                hist = self.node2hist[s_node][0:t_idx]
        else:
            hist = self.node2hist[s_node][t_idx - self.hist_len:t_idx]



        hist_poi_cat = []
        for i in hist:
            poi_id = i[0]
            poi_cat_id = poi_cat[poi_id]
            hist_poi_cat.append(poi_cat_id)


        hist_poi_name = []
        for i in hist:  # 获得历史序列的poiid
            hist_poi_name.append(item2index[i[0]])  # 将poi_id对应的名字添加到hist_poi_name列表中

        hist_poi_weight = []
        for i in hist_poi_name:
            hist_poi_weight.append(poi_weight[i])

        # 这里将大于1的poi频率权重设置为1.
        for i in range(len(hist_poi_weight)):
            if hist_poi_weight[i] > 1:
                hist_poi_weight[i] = 1

        hist_poi_weight = torch.Tensor(hist_poi_weight)

        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]

        # 用于获得历史访问poi的所属的类别序号
        np_h_cat = np.zeros((self.hist_len,))
        np_h_cat[:len(hist_nodes)] = hist_poi_cat

        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_weight = np.zeros((self.hist_len,))
        np_h_weight[:len(hist_nodes)] = hist_poi_weight
        np_h_masks = np.zeros((self.hist_len,))  # 返回一个hist_len大小的0向量
        np_h_masks[:len(hist_nodes)] = 1.  # 将其所有的值设为1

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_cat':t_node_cat,
            'history_nodes': np_h_nodes,
            'history_times': np_h_times,

            'history_masks': np_h_masks,
            'history_cat': np_h_cat,
            'history_frequent_weight':np_h_weight
        }

        return sample
