# 用于改变用户、权重的嵌入维度
embedding = 128
emb_size = 128  # 嵌入的维度为128维
neg_size = 20  # 负样本为10
hist_len = 10
cuda = "0"  # 指定gpu处理
learning_rate = 0.0001

decay = 0.01  # l2正则化衰减率，这个正则化率为音乐推荐的衰减率

# decay = 0.001

batch_size = 128  # 一次处理的数据
test_and_save_step = 20
epoch_num = 180  # 训练周期为100次