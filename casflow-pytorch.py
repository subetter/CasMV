import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class casflow(torch.nn.Module):
    def __init__(self, emb_dim, rnn_units):
        super(casflow, self).__init__()
        self.batch_norm = nn.LayerNorm(emb_dim)
        self.gru_1 = nn.GRU(emb_dim, rnn_units * 2, bidirectional=True, batch_first=True)
        self.gru_2 = nn.GRU(rnn_units * 4, rnn_units, bidirectional=True, batch_first=True)
        self.mlp_1 = nn.Linear(rnn_units, 128)
        self.mlp_2 = nn.Linear(128, 64)
        self.mlp_3 = nn.Linear(64, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * rnn_units, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, cas_inputs, label):
        for i in range(len(cas_inputs)):
            cas_inputs[i] = self.batch_norm(cas_inputs[i])
        padded_input = pad_sequence(cas_inputs, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in cas_inputs], dtype=torch.int64)
        # 0可以替换为其他你希望的填充值
        packed_input = pack_padded_sequence(padded_input, lengths=lengths, batch_first=True, enforce_sorted=False)
        gru_1, _ = self.gru_1(packed_input)
        gru_2, _ = self.gru_2(gru_1)

        # 将填充后的序列进行 pack
        gru_2_output, _ = pad_packed_sequence(gru_2, batch_first=True)
        # 连接前向和后向的输出
        indices = _ - 1
        gru_2_out = gru_2_output[torch.arange(gru_2_output.size(0)), indices, :].detach().clone()

        outputs = self.mlp(gru_2_out)
        loss = F.mse_loss(outputs, label)
        return outputs, loss


max_seq = 100
emb_dim = 80
rnn_units = 128

input = './dataset/twitter/1d/'
with open(input + 'train.pkl', 'rb') as ftrain:
    train_cascade, train_global, train_label = pickle.load(ftrain)
with open(input + 'val.pkl', 'rb') as fval:
    val_cascade, val_global, val_label = pickle.load(fval)
with open(input + 'test.pkl', 'rb') as ftest:
    test_cascade, test_global, test_label = pickle.load(ftest)

model = casflow(emb_dim, rnn_units)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # test_loss: 5.7269503072250725
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)   # test_loss: 5.9404525590497395
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)     # test_loss: 5.819516065508821

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


def pro_cascade(train_cascade, train_global):
    for a in range(len(train_cascade)):
        for i in range(len(train_cascade[a])):
            # 将 i 转成 tensor
            train_cascade[a][i] = torch.tensor(train_cascade[a][i], dtype=torch.float32)

        # 显式将所有张量转换为相同的数据类型
        common_dtype = torch.float32
        train_cascade[a] = [tensor.to(common_dtype) for tensor in train_cascade[a]]

        # 使用 torch.stack 将它们堆叠在一起
        train_cascade[a] = torch.stack(train_cascade[a])

    for a in range(len(train_global)):
        for i in range(len(train_global[a])):
            # 将 i 转成 tensor
            train_global[a][i] = torch.tensor(train_global[a][i], dtype=torch.float32)

        # 显式将所有张量转换为相同的数据类型
        common_dtype = torch.float32
        train_global[a] = [tensor.to(common_dtype) for tensor in train_global[a]]

        # 使用 torch.stack 将它们堆叠在一起
        train_global[a] = torch.stack(train_global[a])

    # 将train_cascade和train_global拼接起来
    cas_input = []
    for a in range(len(train_cascade)):
        cas_input.append(torch.cat((train_cascade[a], train_global[a]), dim=1).to(device))

    return cas_input


cas_input1 = pro_cascade(train_cascade, train_global)
cas_input2 = pro_cascade(val_cascade, val_global)
cas_input3 = pro_cascade(test_cascade, test_global)

batch_size = 64
best_val_loss = 100
train_label = torch.tensor(train_label, dtype=torch.float32).to(device).detach().clone()
train_label[train_label < 1] = 1
train_label = torch.log2(train_label)

label2 = torch.tensor(val_label, dtype=torch.float32).to(device).detach().clone()
label2[label2 < 1] = 1
label2 = torch.log2(label2)

for epoch in range(300):
    model.train()
    # 分批次训练
    loss1 = []
    for i in range(math.ceil(len(cas_input1) / batch_size)):
        batch = cas_input1[i * batch_size:(i + 1) * batch_size]
        label = train_label[i * batch_size:(i + 1) * batch_size].detach().clone()
        label = label.reshape(-1, 1)
        label = label.to(device)
        out, loss = model(batch, label)
        # label < 1的赋值为1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss1.append(loss.item())
    print("train_loss: ", np.mean(loss1), end="   ")

    loss2 = []
    for i in range(math.ceil(len(cas_input2) / batch_size)):
        model.eval()
        batch = cas_input2[i * batch_size:(i + 1) * batch_size]
        label = label2[i * batch_size:(i + 1) * batch_size].detach().clone()
        label = label.reshape(-1, 1)
        label = label.to(device)
        out, loss = model(batch, label)
        loss2.append(loss.item())
    print("val_loss:", np.mean(loss2))
    if (np.mean(loss2) < best_val_loss):
        best_val_loss = np.mean(loss2)
        torch.save(model.state_dict(), './twitter-model1.pth')

# 加载保存好的模型
model.load_state_dict(torch.load('twitter-model1.pth'))
label3 = torch.tensor(test_label, dtype=torch.float32).to(device).detach().clone()
label3[label3 < 1] = 1
label3 = torch.log2(label3)
loss3 = []
model.eval()
for i in range(math.ceil(len(cas_input3) / batch_size)):
    batch = cas_input3[i * batch_size:(i + 1) * batch_size]
    label = label3[i * batch_size:(i + 1) * batch_size].detach().clone()
    label = label.reshape(-1, 1)
    label = label.to(device)
    out, loss = model(batch, label)
    loss3.append(loss.item())
print("test_loss:", np.mean(loss3))
