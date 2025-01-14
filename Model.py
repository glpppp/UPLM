# -*- coding: utf-8 -*-
import pickle

import torch
from haversine import haversine
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from Data import DataSetTrain, DataSetTestNext
import os
import logging

FType = torch.FloatTensor  # 转换为float型tensor
LType = torch.LongTensor  # 转换为long型tensor

FORMAT = "%(asctime)s - %(message)s"  # 定义日志格式。asctime为时间。message为info中的内容
logging.basicConfig(level=logging.INFO, format=FORMAT)


class ATPP:  # 定义ATPP类
    def __init__(self, dataset_name: object, file_path_tr: object, file_path_te: object,
                 emb_size: object = 128,
                 neg_size: object = 10, hist_len: object = 2, embedding: object=0 ,user_count: object = 992, item_count: object = 5000,cat_count :object=100,
                 directed: object = True, learning_rate: object = 0.001,
                 decay: object = 0.001, batch_size: object = 1024, test_and_save_step: object = 50,
                 epoch_num: object = 100, top_n: object = 30, sample_time: object = 3,
                 sample_size: object = 100, use_hist_attention: object = True, use_poi_frequent: object = True,
                 use_corr_matrix: object = True,
                 num_workers: object = 0,
                 norm_method: object = 'hour') -> object:  # 定义实例属性
        self.dataset_name = dataset_name
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.embedding = embedding
        self.user_count = user_count
        self.item_count = item_count
        self.cat_count = cat_count

        self.lr = learning_rate
        self.decay = decay
        self.batch = batch_size
        self.test_and_save_step = test_and_save_step
        self.epochs = epoch_num

        self.top_n = top_n
        self.sample_time = sample_time
        self.sample_size = sample_size

        self.directed = directed
        self.use_hist_attention = use_hist_attention

        self.use_poi_frequent = use_poi_frequent
        self.use_corr_matrix = use_corr_matrix

        self.num_workers = num_workers
        self.norm_method = norm_method
        self.is_debug = False

        self.data_tr = DataSetTrain(file_path_tr, user_count=self.user_count,
                                    item_count=self.item_count,
                                    neg_size=self.neg_size, hist_len=self.hist_len, directed=self.directed)  # 实例化训练集类
        self.data_te_old = DataSetTestNext(file_path_te, self.data_tr, user_count=self.user_count,
                                           item_count=self.item_count,
                                           hist_len=self.hist_len, directed=self.directed)  # 实例化测试集类

        self.node_dim = self.data_tr.get_node_dim()  # 将训练集中的node_dim返回。user_count+item_count.
        self.node_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.node_dim, self.emb_size)),
            dtype=torch.float)  # random.uniform随机生成size个[-0.5 / self.emb_size, 0.5 / self.emb_size)范围的数，返回一个size大小的张量。这里为node_dim行，emb_size列。左闭右开。这里emb_size为128。这里就是先生成每一个用户与歌曲的128维随机张量。


        self.user_node_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.node_dim, self.emb_size+self.embedding)),
            dtype=torch.float)  # random.uniform随机生成size个[-0.5 / self.emb_size, 0.5 / self.emb_size)范围的数，返回一个size大小的张量。这里为node_dim行，emb_size列。左闭右开。这里emb_size为128。这里就是先生成每一个用户与歌曲的128维随机张量。


# 对poi所属类别的序号进行嵌入表示
        self.cat_emb = torch.tensor(
            np.random.uniform(-0.5 / self.emb_size, 0.5 / self.emb_size, size=(self.cat_count, self.emb_size)),
            dtype=torch.float)


        self.use_time_long = True
        self.use_time_short =True

        self.delta_interval_weight = torch.ones(self.node_dim, dtype=torch.float)  # 返回一个node_dim大小的1向量

        self.delta_duration_weight = torch.ones(self.node_dim, dtype=torch.float)  # 生成一个node_dim大小的1向量
        self.hist_attention_weight_long = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.hist_len - 1, self.emb_size+self.embedding)),
            dtype=torch.float)  # normal正态分布。生成一个以0为中心，以np.sqrt为中心的值，返回一个size大小的张量。行为hist_len，列为emb_size.
        #  生成size个符合均值为0，标准差为np.sqrt(2/128)标准正态分布的概率密度随机数。作为长期偏好权重
        self.hist_attention_weight_short = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(2, self.emb_size+self.embedding)), dtype=torch.float)  # 同上。作为短期偏好权重。权重矩阵大小为2*emb_size
        self.corr_matrix = torch.tensor(
            np.random.normal(0, np.sqrt(2.0 / self.emb_size), size=(self.emb_size+self.embedding, self.emb_size+self.embedding)),  # 核矩阵大小为emb_size*emb_size
            dtype=torch.float)


        self.weight_poi = torch.ones(self.emb_size,self.emb_size+self.embedding)
        self.weight_cat = torch.ones(self.emb_size,self.emb_size+self.embedding)


        if torch.cuda.is_available():  # 如果gpu存在，执行下列代码。将各个属性放到gpu上进行计算
            self.node_emb = self.node_emb.cuda()  # 将node_emb 等于node_emb.cuda().将数据放到gpu上进行计算
            self.cat_emb = self.cat_emb.cuda()

            self.delta_interval_weight = self.delta_interval_weight.cuda()
            self.hist_attention_weight_long = self.hist_attention_weight_long.cuda()
            self.hist_attention_weight_short = self.hist_attention_weight_short.cuda()
            self.corr_matrix = self.corr_matrix.cuda()
            self.delta_duration_weight = self.delta_duration_weight.cuda()
        self.node_emb.requires_grad = True  # requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度
        self.cat_emb.requires_grad = True

        self.delta_interval_weight.requires_grad = True
        self.hist_attention_weight_long.requires_grad = True
        self.hist_attention_weight_short.requires_grad = True
        self.corr_matrix.requires_grad = True
        self.delta_duration_weight.requires_grad = True
        self.weight_poi.requires_grad = True
        self.weight_cat.requires_grad = True

        self.use_cat_pre = True

        self.opt = Adam(lr=self.lr,
                        params=[self.weight_cat,self.weight_poi,self.user_node_emb,self.node_emb,self.cat_emb, self.delta_interval_weight, self.hist_attention_weight_long,
                                self.hist_attention_weight_short,
                                self.corr_matrix, self.delta_duration_weight], weight_decay=self.decay)  # 实例化Adam优化器类.lr为学习率，params为待优化的项，weight_decay为l2正则化参数
        self.loss = torch.FloatTensor()  # 定义一个空的float类型的tensor

        self.lstm_layer = nn.LSTM(input_size=3*self.emb_size,hidden_size=self.emb_size,num_layers=1,dropout=0,batch_first=True)

    def forward(self, s_nodes, t_nodes, t_times, t_cat,n_nodes, h_nodes, h_times, h_time_mask,h_cat,neg_poi_cat,history_frequent_weight):  # 前向传播函数.t_times是用户倾听记录中某一首歌曲距离记录中第一首歌的时间

        batch = s_nodes.size()[0]  # 为了得到这个张量的大小。大小为batch_size。。。先获取有多少人


        s_node_emb = torch.index_select(self.user_node_emb, 0, s_nodes.view(-1)).view(batch, -1)  # 将这几个属性特征转换为node_embz维度的张量。得到嵌入层。意思就是：index_select就是取node_emb张量中，第0个维度且索引号为s_nodes.view(-1)=batch_size的1x128的张量子集。

        #s_node_emb = torch.index_select(self.node_emb, 0, s_nodes.view(-1)).view(batch, -1)
        t_node_emb = torch.index_select(self.node_emb, 0, t_nodes.view(-1)).view(batch, -1)  # 然后再使用view方法，将结果转换为batch行，列为-1（系统自行选择）的张量

        # 用于得到目标poi所属类别的序号嵌入表示
        t_cat_node_emb = torch.index_select(self.cat_emb,0,t_cat.view(-1)).view(batch,-1)

        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).view(batch, self.hist_len, -1)  # h_nodes是一个二维的张量，先用view（-1）将其转换成将结果转换为一阶张量，然后去node_emb张量中，对应索引号的张量子集。、

        # 用于得到历史访问poi的所属类别的序号嵌入表示
        h_node_cat_emb = torch.index_select(self.cat_emb,0,h_cat.view(-1)).view(batch,self.hist_len,-1)

        h_node_emb_copy = torch.cat((h_node_emb,h_node_cat_emb),2)


        n_node_emb = torch.index_select(self.node_emb, 0, n_nodes.view(-1)).view(batch, self.neg_size, -1)  #最后将得到的结果转换为batch*hist_len*-1（系统随机选择） = 张量子集中的数 的张量


        n_node_cat_emb = torch.index_select(self.cat_emb,0,neg_poi_cat.view(-1)).view(batch,self.neg_size,-1)
        new_n_node_emb = torch.cat((n_node_emb,n_node_cat_emb),2)
        ttp_long = 1
        ttp_short = 1


        self.delta_interval_weight.data.clamp_(min=1e-6)  # 在原来属性的基础上将最小值全部转化为1，最小值为0.000001.其实delta_interval_weight还是保持不变
        # 选择delta_interval_weight中与用户序号相对应的权重。
        delta_interval_weight = torch.index_select(self.delta_interval_weight, 0, s_nodes.view(-1)).unsqueeze(1)  # 对于delta_interval_weight，取第0维且索引号为s_nodes对应值的张量子集。然后将1*n张量结果转化为n*1张量
        time_interval = torch.abs(t_times.unsqueeze(1) - h_times)  # 将距离第一首歌的时间1*n张量转换为n*1张量，然后将结果减去hist_times。将每一首目标音乐的时间减去hist列表中的音乐时间。得到每一首目标音乐与hist列表中的音乐时间差。转换为绝对值形式

        # time_duration_ratio是指的用户行为，如果有跳过行为，则为一个数，如果没有跳过行为，则ratio值为1
        time_duration_ratio = torch.ones((batch, self.hist_len), dtype=torch.float)  # 生成一个batch*hist_len的全1张量
        if torch.cuda.is_available():
            time_duration_ratio = time_duration_ratio.cuda()  # 在gpu上进行计算
        # if self.use_duration:  # 如果为TRUE，执行下列代码    这一步实际上就是为了进行消融实验，当不利用用户行为信息的时候，结果如何。
        if self.use_poi_frequent:  # 如果使用poi_frquent的话，执行下列代码
            self.delta_duration_weight.data.clamp_(min=1e-6)
            delta_duration_weight = torch.index_select(self.delta_duration_weight, 0, s_nodes.view(-1)).unsqueeze(
                     1).expand(batch, self.hist_len)
            time_duration_ratio = delta_duration_weight*history_frequent_weight

        h_index = self.hist_len - 1  # 将hist_len - 1 赋值给h_index。是因为张量下标从0开始


        if self.use_time_long == True:
            ttp_long = torch.exp(torch.neg(delta_interval_weight) * time_interval[:, :h_index])

        if self.use_time_short == True:
            ttp_short = torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, h_index].unsqueeze(1))


        if self.use_hist_attention:  # 如果use_hist_attention为TRUE的话，执行下列代码

            temp_product_long = torch.mul(h_node_emb_copy[:, :h_index, :], self.hist_attention_weight_long.unsqueeze(0))

            attention_long = softmax(((s_node_emb.unsqueeze(1) - temp_product_long) ** 2).sum(dim=2).neg(), dim=1)  # 将s_node_emb张量减去temp_product_long张量，然后平方处理，之后将张量进行sum处理，再求其相反数。

            aggre_hist_node_emb = (attention_long.unsqueeze(2) * h_node_emb_copy[:, :h_index, :] *
            #aggre_hist_node_emb = ( attention_long.unsqueeze(2) * h_node_emb[:, :h_index, :] *
                                   (time_duration_ratio[:, :h_index] * ttp_long * h_time_mask[:, :h_index])
                                    .unsqueeze(2)).sum(dim=1)  # 这个结果得到的是

            curr_node_emb = h_node_emb_copy[:, h_index, :] * (
            #curr_node_emb = h_node_emb[:, h_index, :] * (
                    time_duration_ratio[:, h_index].unsqueeze(1) * ttp_short * h_time_mask[:,
                                                                                             h_index].unsqueeze(1))
            new_h_node_emb = torch.cat([aggre_hist_node_emb.unsqueeze(1), curr_node_emb.unsqueeze(1)], dim=1)  # torch.cat方法，将两个张量拼接在一起

            # 这一步相当于计算短期偏好权重中，Ws * Vmn
            temp_product_short = torch.mul(new_h_node_emb, self.hist_attention_weight_short.unsqueeze(0))
            # 短期偏好权重的计算结果
            attention_short = softmax(((s_node_emb.unsqueeze(1) - temp_product_short) ** 2).sum(dim=2).neg(), dim=1)  # 得到的结果是短期偏好的权重

            pref_embedding = (attention_short.unsqueeze(2) * new_h_node_emb).sum(dim=1)

        else:
            pref_embedding = (h_node_emb_copy * (time_duration_ratio * torch.exp(
            #pref_embedding = (h_node_emb * (time_duration_ratio * torch.exp(
                torch.neg(delta_interval_weight) * time_interval) * h_time_mask).unsqueeze(2)).sum(dim=1) / (
                                     self.hist_len * 1.)
        if self.use_corr_matrix:  # 如果使用核矩阵的话，执行下列代码

            new_pref_embedding = torch.matmul(pref_embedding, self.corr_matrix)
        else:
            new_pref_embedding = pref_embedding

        new_t_node_emb = torch.cat((t_node_emb, t_cat_node_emb), 1)

        if self.use_cat_pre:
            cat_pre = self.get_output1(s_node_emb,h_node_cat_emb,self.lstm_layer)


            p_lambda = ((new_pref_embedding - new_t_node_emb) ** 2).sum(dim=1).neg()
            p_cat_lambda = ((cat_pre - t_cat_node_emb)**2).sum(dim=1).neg()
            p_lambda_new = p_lambda + p_cat_lambda

            n_lambda = ((new_pref_embedding.unsqueeze(1) - new_n_node_emb) ** 2).sum(dim=2).neg()
            # n_lambda = ((new_pref_embedding.unsqueeze(1) - n_node_emb) ** 2).sum(dim=2).neg()
            n_cat_lambda = ((cat_pre.unsqueeze(1) - n_node_cat_emb) ** 2).sum(dim=2).neg()
            n_lambda_new = n_cat_lambda + n_lambda

        else:
            p_lambda_new = ((new_pref_embedding - new_t_node_emb) ** 2).sum(dim=1).neg()

            n_lambda_new = ((new_pref_embedding.unsqueeze(1) - new_n_node_emb) ** 2).sum(dim=2).neg()

        return p_lambda_new, n_lambda_new

    def get_output1(self, s_node_emb, cat_node_emb, lstm_layer):

        s_node_emb = s_node_emb.unsqueeze(1).repeat(1,10,1)
        input_tensor = torch.cat((s_node_emb, cat_node_emb), 2)

        output, _ = lstm_layer(input_tensor)

        end = output[:,-1,:]
        return end

    def loss_func(self, s_nodes, t_nodes, t_times,t_cat, n_nodes, h_nodes, h_times,
                  h_time_mask,h_cat,neg_poi_cat,history_frequent_weight):

        p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, t_cat,n_nodes, h_nodes, h_times,h_time_mask,h_cat,neg_poi_cat,history_frequent_weight)

        loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) \
               - torch.log(torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)  # 负样本损失函数的计算方法

        return loss

    def update(self, s_nodes, t_nodes, t_times, t_cat,n_nodes, h_nodes, h_times,
               h_time_mask,h_cat,neg_poi_cat,history_frequent_weight):
        self.opt.zero_grad()

        loss = self.loss_func(s_nodes, t_nodes, t_times,t_cat, n_nodes, h_nodes, h_times,
                              h_time_mask,h_cat,neg_poi_cat,history_frequent_weight)
        loss = loss.sum()  # 对损失值进行累加
        self.loss += loss.data  # 将self.loss加上loss的值。0 + loss.data
        loss.backward()  # 损失函数的反向传播。。同时计算所有参数的梯度
        self.opt.step()   # 对参数进行优化

    def train_and_test(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            if os.name == 'nt':  # 表示的是如果为Windows操作系统，执行下列代码
                loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=0,drop_last=True)  # 加载训练集中的数据，一次加载batch个数据.DataLoader加载的数据为张量类型
            else:  # 否则，执行下列代码
                loader = DataLoader(self.data_tr, batch_size=self.batch, shuffle=True, num_workers=self.num_workers,drop_last=True)
            for i_batch, sample_batched in enumerate(loader):  # 对数据进行枚举
                self.is_debug = False

                if torch.cuda.is_available():
                    self.update(sample_batched['source_node'].type(LType).cuda(),
                                sample_batched['target_node'].type(LType).cuda(),
                                sample_batched['target_time'].type(FType).cuda(),
                                sample_batched['target_cat'].type(LType).cuda(),
                                sample_batched['neg_nodes'].type(LType).cuda(),
                                sample_batched['history_nodes'].type(LType).cuda(),
                                sample_batched['history_times'].type(FType).cuda(),
                                sample_batched['history_masks'].type(FType).cuda(),
                                sample_batched['history_cat'].type(LType).cuda(),
                                sample_batched['neg_node_cat'].type(LType).cuda(),
                                sample_batched['history_frequent_weight'].type(FType).cuda()
                                )

                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['target_cat'].type(LType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType),
                                sample_batched['history_cat'].type(LType),
                                sample_batched['neg_node_cat'].type(LType),
                                sample_batched['history_frequent_weight'].type(FType)
                                )

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data_tr)) + '\n')  # 将结果进行输出
            sys.stdout.flush()  # 对输出通道进行清洗
            if ((epoch + 1) % self.test_and_save_step == 0) or epoch == 0 or epoch == 4 or epoch == 150  or epoch == 165 or epoch == 170 or epoch == 175:  # 符合此条件的epoch，执行下列方法
                self.recommend(epoch, is_new_item=False)

    def recommend(self, epoch, is_new_item=False):
        count_all = 0
        rate_all_sum = 0
        recall_all_sum = np.zeros(self.top_n)  # 返回一个top_n=30大小的全0向量，用于存储recall率
        MRR_all_sum = np.zeros(self.top_n)  # 返回一个top_n=30大小的全0向量，用于存储MRR率

        if is_new_item:  # 如果is_new_item为TRUE，执行下列代码
            loader = DataLoader(self.data_te_new, batch_size=self.batch, shuffle=False, num_workers=self.num_workers,drop_last=True)  # 加载data_te_new实例。做next new recommendation
        else:  # 否则执行下列代码，
            loader = DataLoader(self.data_te_old, batch_size=self.batch, shuffle=False, num_workers=self.num_workers,drop_last=True)  # 加载data_te_old实例。做next 推荐
        for i_batch, sample_batched in enumerate(loader):
            if torch.cuda.is_available():
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType).cuda(),  # 调用evaluate方法，计算召回率，MRR率
                                  sample_batched['target_node'].type(LType).cuda(),
                                  sample_batched['target_time'].type(FType).cuda(),
                                  sample_batched['target_cat'].type(LType).cuda(),
                                  sample_batched['history_nodes'].type(LType).cuda(),
                                  sample_batched['history_times'].type(FType).cuda(),
                                  sample_batched['history_masks'].type(FType).cuda(),
                                  sample_batched['history_cat'].type(LType).cuda(),
                                  sample_batched['history_frequent_weight'].type(FType).cuda()
                                  )
            else:
                rate_all, recall_all, MRR_all = \
                    self.evaluate(sample_batched['source_node'].type(LType),
                                  sample_batched['target_node'].type(LType),
                                  sample_batched['target_time'].type(FType),
                                  sample_batched['target_cat'].type(LType),
                                  sample_batched['history_nodes'].type(LType),
                                  sample_batched['history_times'].type(FType),
                                  sample_batched['history_masks'].type(FType),
                                  sample_batched['history_cat'].type(LType),
                                  sample_batched['history_frequent_weight'].type(FType))
            count_all += self.batch
            rate_all_sum += rate_all
            recall_all_sum += recall_all
            MRR_all_sum += MRR_all

        rate_all_sum_avg = rate_all_sum * 1. / count_all
        recall_all_avg = recall_all_sum * 1. / count_all
        MRR_all_avg = MRR_all_sum * 1. / count_all
        if is_new_item:
            logging.info('=========== testing next new item epoch: {} ==========='.format(epoch))
            logging.info('count_all_next_new: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next_new: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next_new: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next_new: {}'.format(MRR_all_avg))
        else:
            logging.info('~~~~~~~~~~~~~ testing next item epoch: {} ~~~~~~~~~~~~~'.format(epoch))
            logging.info('count_all_next: {}'.format(count_all))
            logging.info('rate_all_sum_avg_next: {}'.format(rate_all_sum_avg))
            logging.info('recall_all_avg_next: {}'.format(recall_all_avg))
            logging.info('MRR_all_avg_next: {}'.format(MRR_all_avg))


    def evaluate(self, s_nodes, t_nodes, t_times, t_cat,h_nodes, h_times, h_time_mask,h_cat,history_frequent_weight):

        path = open('../poi_cat_dict_path', 'rb')
        poi_cat_data = pickle.load(path)

        poi_location = pickle.load(open('./POI_location','rb'))

        s_node_current = list()
        for i in range(len(h_nodes)):
            current = h_nodes[i][-1]
            s_node_current.append(current)

        poi_candidate_list = []

        for i in range(len(s_node_current)):
            candidate = []
            user_poi_id = int(s_node_current[i])
            user_poi_id_lon, user_poi_id_lat = poi_location[user_poi_id]
            locate = (int(user_poi_id_lon), int(user_poi_id_lat))

            for j, (other_poi_lon, other_poi_lat) in enumerate(poi_location):
                other_poi_lon = int(other_poi_lon)
                other_poi_lat = int(other_poi_lat)
                other_poi = (other_poi_lon, other_poi_lat)
                distance = haversine(locate, other_poi)
                if distance <= 1000:
                    candidate.append(j)

            poi_candidate_list.append(candidate)

        poi_candidate_cat_list = []

        for user_can in poi_candidate_list:
            candidate_cat = [poi_cat_data.get(can) for can in user_can]
            poi_candidate_cat_list.append(candidate_cat)


        batch = s_nodes.size()[0]  # 首先获取batch_size的大小
        all_item_index = torch.arange(0, self.item_count)  # 生成0到item_count（66407）的数，从0到66406。tensor数据类型
        if torch.cuda.is_available():
            all_item_index = all_item_index.cuda()

        cat_list = []
        for i in range(len(all_item_index)):
            poi_cat_id = poi_cat_data[i]
            cat_list.append(poi_cat_id)
        cat_list_tensor = torch.Tensor(cat_list).long()

        all_item_cat_index = torch.arange(0, self.cat_count)
        if torch.cuda.is_available():
            all_item_cat_index = all_item_index.cuda()

        all_cat_emb = torch.index_select(self.cat_emb,0,cat_list_tensor).detach()

        all_node_emb = torch.index_select(self.node_emb, 0, all_item_index).detach()  # 先获得node_emb中所有all_item_index的嵌入表示，然后生成一个新的张量

        new_all_node_emb = torch.cat((all_node_emb,all_cat_emb),1)


        h_node_emb = torch.index_select(self.node_emb, 0, h_nodes.view(-1)).detach().view(batch, self.hist_len, -1)  # 获得h_node_emb的嵌入表示

        h_node_cat_emb = torch.index_select(self.cat_emb,0,h_cat.view(-1)).detach().view(batch,self.hist_len,-1)

        h_node_emb_copy = torch.cat((h_node_emb,h_node_cat_emb),2)

        ttp_long = 1
        ttp_short = 1


        self.delta_interval_weight.data.clamp_(min=1e-6)
        time_interval = torch.abs(t_times.unsqueeze(1) - h_times)  # 得到目标音乐倾听时间与历史音乐倾听时间的时间差
        delta_interval_weight = torch.index_select(self.delta_interval_weight, 0, s_nodes.view(-1)).detach().unsqueeze(
            1)

        s_node_emb = torch.index_select(self.user_node_emb, 0, s_nodes.view(-1)).detach().view(batch, -1)
        time_duration_ratio = torch.ones((batch, self.hist_len), dtype=torch.float)
        if torch.cuda.is_available():
            time_duration_ratio = time_duration_ratio.cuda()

        if self.use_poi_frequent:
            self.delta_duration_weight.data.clamp_(min=1e-6)
            delta_duration_weight = torch.index_select(self.delta_duration_weight, 0, s_nodes.view(-1)).unsqueeze(
                     1).expand(batch, self.hist_len)
            time_duration_ratio = delta_duration_weight*history_frequent_weight
        h_index = self.hist_len - 1
        if self.use_time_long == True:
            ttp_long = torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, :h_index])
        if self.use_time_short == True:
            ttp_short = torch.exp(
                torch.neg(delta_interval_weight) * time_interval[:, h_index].unsqueeze(1))

        if self.use_hist_attention:
            temp_product_long = torch.mul(h_node_emb_copy[:, :h_index, :],
                                          self.hist_attention_weight_long.detach().unsqueeze(0))
            attention_long = softmax(((s_node_emb.unsqueeze(1) - temp_product_long) ** 2).sum(dim=2).neg(), dim=1)

            aggre_hist_node_emb = (attention_long.unsqueeze(2) * h_node_emb_copy[:, :h_index, :] * (
                    time_duration_ratio[:, :h_index] * ttp_long * h_time_mask[:, :h_index]).unsqueeze(
                2)).sum(dim=1)

            curr_emb = h_node_emb_copy[:, h_index, :] * (
                    time_duration_ratio[:, h_index].unsqueeze(1) * ttp_short * h_time_mask[:,
                                                                                             h_index].unsqueeze(1))
            new_h_node_emb = torch.cat([aggre_hist_node_emb.unsqueeze(1), curr_emb.unsqueeze(1)], dim=1)
            temp_product_short = torch.mul(new_h_node_emb, self.hist_attention_weight_short.detach().unsqueeze(0))
            attention_short = softmax(((s_node_emb.unsqueeze(1) - temp_product_short) ** 2).sum(dim=2).neg(), dim=1)
            pref_embedding = (attention_short.unsqueeze(2) * new_h_node_emb).sum(dim=1)
        else:
            pref_embedding = (h_node_emb_copy * (time_duration_ratio * torch.exp(

                torch.neg(delta_interval_weight) * time_interval) * h_time_mask).unsqueeze(2)).sum(dim=1) / (
                                     self.hist_len * 1.)
        if self.use_corr_matrix:
            new_pref_embedding = torch.matmul(pref_embedding, self.corr_matrix.detach())
        else:
            new_pref_embedding = pref_embedding  #1024*128阶张量，作为用户的偏好

        cat_pre = self.get_output1(s_node_emb, h_node_cat_emb, self.lstm_layer)


        new_cat_pref_embedding_norm = (cat_pre**2).sum(1).view(batch,1)

        new_pref_embedding_norm = (new_pref_embedding ** 2).sum(1).view(batch, 1)  # 1024 *1 阶张量.将每一个偏好的嵌入表示作平方和
        p_lambda = torch.zeros(batch,self.item_count)
        for i in range(batch):
            candidate_embedding = torch.zeros(self.item_count,self.emb_size+self.embedding)
            candidate_cat_embedding = torch.index_select(self.cat_emb, 0,all_item_cat_index).detach()

            user_candidate = poi_candidate_list[i]

            user_candidate_copy = torch.zeros(len(user_candidate),dtype=torch.int32)


            for j in range(len(user_candidate)):
                user_candidate_copy[j] = int(user_candidate[j])

            user_candidate_embedding =torch.index_select(new_all_node_emb,0,user_candidate_copy).detach()
            for j in range(len(user_candidate_embedding)):

                candidate_index = user_candidate[j]

                candidate_embedding[candidate_index] = user_candidate_embedding[j]



            all_node_emb_norm1 = (candidate_embedding ** 2).sum(1).view(1, self.item_count)

            all_cat_emb_norm1 = (candidate_cat_embedding**2).sum(1).view(1,self.cat_count)


            new_pref_embedding_norm1 = new_pref_embedding_norm[i]
            new_cat_pref_embedding_norm1 = new_cat_pref_embedding_norm[i]
            # 获得当前用户的类别偏好向量
            cat_pre1 = cat_pre[i]

            # 获得当前用户的poi偏好向量
            new_pref_embedding1 = new_pref_embedding[i]

            p_lambda1 = (new_pref_embedding_norm1 + all_node_emb_norm1 - 2.0 * torch.matmul(new_pref_embedding1,
                                                                                     torch.transpose(candidate_embedding, 0, 1))).neg()


            p_cat_lambd = (new_cat_pref_embedding_norm1 + all_cat_emb_norm1 - 2.0 * torch.matmul(cat_pre1,torch.transpose(candidate_cat_embedding,0,1))).neg()

            if self.use_cat_pre:
                p_cat_lamda = torch.zeros(1, len(cat_list))

                # 使用索引和广播进行一次性赋值
                p_cat_lamda[:, :] = p_cat_lambd[:, cat_list]

                p_lambda_end = p_lambda1 + p_cat_lamda
                p_lambda[i] = p_lambda_end
            else:

                p_lambda[i] =p_lambda1
        rate_all_sum = 0
        recall_all = np.zeros(self.top_n)

        MRR_all = np.zeros(self.top_n)

        t_nodes_list = t_nodes.cpu().numpy().tolist()

        p_lambda_numpy = p_lambda.cpu().detach().numpy()

        for i in range(len(t_nodes_list)):
            t_node = t_nodes_list[i]

            p_lambda_numpy_i_item = p_lambda_numpy[i]  #
            prob_index = np.argsort(-p_lambda_numpy_i_item).tolist()
            gnd_rate = prob_index.index(t_node) + 1
            rate_all_sum += gnd_rate
            if gnd_rate <= self.top_n:
                recall_all[gnd_rate - 1:] += 1
                MRR_all[gnd_rate - 1:] += 1. / gnd_rate

        return rate_all_sum, recall_all, MRR_all
