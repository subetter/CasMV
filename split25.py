import pickle as pk
from collections import Counter
from matplotlib import pyplot as plt
import datetime
# 读取txt文件
weibo_less25 = []
weibo_more25 = []
label_data = []
with open('dataset/twitter/dataset.txt', 'r') as f:
    weibo = f.readlines()
for line in weibo:
    # line = line.split('\t')
    # label_data.append(int(line[3]))
    if(int(line.split('\t')[3]) <= 25):
        weibo_less25.append(line)
    else:
        weibo_more25.append(line)

# 将数据写入文件
with open('dataset/twitter/less25.txt', 'w') as f:
    for i in weibo_less25:
        f.write(i)

with open('dataset/twitter/more25.txt', 'w') as f:
    for i in weibo_more25:
        f.write(i)

#统计 label中25有多少个
# count = 0
# for i in label_data:
#     if i == 25:
#         count += 1
#
# print(count)

# 前25和后25的