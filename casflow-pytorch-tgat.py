import torch.nn as nn
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from modules.utils import MergeLayer_output, Feat_Process_Layer
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode
import math
import torch
import torch.nn.functional as F
import datasets1 as dataset
import torch.utils.data
import numpy as np
from option import args
from utils import EarlyStopMonitor, logger_config
import datetime, os
from train1 import getcas
from my_utils import msle, male, mape, pcc


class casflow(torch.nn.Module):
    def __init__(self, emb_dim, rnn_units, config):
        super(casflow, self).__init__()
        self.batch_norm = nn.LayerNorm(emb_dim + config.input_dim)
        self.gru_1 = nn.GRU(emb_dim + config.input_dim, rnn_units*2, bidirectional=True, batch_first=True)
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

        # TGAT部分
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.edge_dim = self.cfg.input_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer

        self.mode = self.cfg.mode

        self.time_encoder = TimeEncode(dimension=self.dims)
        self.embedding_module_type = self.cfg.module_type
        self.embedding_module = get_embedding_module(module_type=self.embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     node_features_dims=self.dims,
                                                     edge_features_dims=self.dims,
                                                     time_features_dim=self.dims,
                                                     hidden_dim=self.dims,
                                                     n_heads=self.n_heads, dropout=self.dropout)

        self.node_preocess_fn = Feat_Process_Layer(self.nodes_dim, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)

    def forward(self, cas_inputs, batch, label):
        # TGAT -----> time feature -----> C
        # BiGCN   ---->  D + C --> GCN
        for i in range(len(cas_inputs)):
            source_node_embedding = self.compute_temporal_embeddings(batch[i]["src_neigh_edge"],
                                                                     batch[i]["src_edge_to_time"],
                                                                     batch[i]["src_edge_feature"],
                                                                     batch[i]["src_node_features"])
            root_embedding = source_node_embedding[batch[i]["src_center_node_idx"], :]
            cas_inputs[i] = torch.cat((cas_inputs[i], root_embedding), dim=1).to(device)
            # 取出后80维的特征
            # cas_inputs[i] = cas_inputs[i][:, -80:]
            cas_inputs[i] = self.batch_norm(cas_inputs[i])
        padded_input = pad_sequence(cas_inputs, batch_first=True, padding_value=0)
        lengths = torch.tensor([len(seq) for seq in cas_inputs], dtype=torch.int64)
        # 0 可以替换为其他你希望的填充值
        packed_input = pack_padded_sequence(padded_input, lengths=lengths, batch_first=True, enforce_sorted=False)
        gru_1, _ = self.gru_1(packed_input)
        gru_2, _ = self.gru_2(gru_1)

        # 将填充后的序列进行 pack
        gru_2_output, _ = pad_packed_sequence(gru_2, batch_first=True)
        # 连接前向和后向的输出
        # gru_2_output_concatenated = torch.cat((gru_2_output[:, :, :rnn_units * 2], gru_2_output[:, :, rnn_units * 2:]),
        #                                       dim=-1)
        # 取平均
        # gru_2_output_avg = torch.mean(gru_2_output, dim=1)
        # 取最后一个时刻的
        indices = _ - 1
        # gru_2_output_selected = torch.gather(gru_2_output, dim=1,
        #                                      index=torch.tensor(indices))

        gru_2_out = gru_2_output[torch.arange(gru_2_output.size(0)), indices, :].detach().clone()

        # mlp_1 = self.mlp_1(gru_2_out)
        # mlp_2 = self.mlp_2(mlp_1)
        # outputs = self.mlp_3(mlp_2)
        outputs = self.mlp(gru_2_out)
        loss = F.mse_loss(outputs, label)
        return outputs, loss

    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat):
        node_feat = self.node_preocess_fn(node_feat)
        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat)
        return node_embedding


max_seq = 25
emb_dim = 80
rnn_units = 128

### 加载TGAT所需数据-----------------------------------------------------------------------------------------------------
config = args
# 设置args 的node_dim
config.node_dim = 40
config.edge_dim = 40

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# log file name set
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_base_path = f"{os.getcwd()}/train_log"
file_list = os.listdir(log_base_path)
max_num = [0]  # [int(fl.split("_")[0]) for fl in file_list if len(fl.split("_"))>2] + [-1]
log_base_path = f"{log_base_path}/{max(max_num) + 1}_{now_time}"
# log and path
get_checkpoint_path = lambda \
        epoch: f'{log_base_path}saved_checkpoints/{args.data_set}-{args.mode}-{args.module_type}-{args.mask_ratio}-{epoch}.pth'
logger = logger_config(log_path=f'{log_base_path}/log.txt', logging_name='gdn')
logger.info(config)

dataset_train = dataset.DygDataset(config, "train")
dataset_valid = dataset.DygDataset(config, "val")
dataset_test = dataset.DygDataset(config, "test")

gpus = None if config.gpus == 0 else config.gpus

max_val_auc, max_test_auc = 0.0, 0.0
early_stopper = EarlyStopMonitor()
best_auc = [0, 0, 0]
all_cas_train, all_cas_val, all_cas_test = [], [], []
for i in range(len(dataset_train)):
    all_cas_train.append(getcas(dataset_train.full_data[i], dataset_train.ngh_finder[i]))
for i in range(len(dataset_valid)):
    all_cas_val.append(getcas(dataset_valid.full_data[i], dataset_valid.ngh_finder[i]))
for i in range(len(dataset_test)):
    all_cas_test.append(getcas(dataset_test.full_data[i], dataset_test.ngh_finder[i]))

### 加载casflow数据-------------------------------------------------------------------------------------------------------
input = './dataset/twitter/2d/'
with open(input + 'train.pkl', 'rb') as ftrain:
    train_cascade, train_global, train_label = pickle.load(ftrain)
with open(input + 'val.pkl', 'rb') as fval:
    val_cascade, val_global, val_label = pickle.load(fval)
with open(input + 'test.pkl', 'rb') as ftest:
    test_cascade, test_global, test_label = pickle.load(ftest)

model = casflow(emb_dim, rnn_units, config)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # test_loss: 5.819516065258821

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


#
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

# 创建EarlyStopMonitor实例
early_stopping = EarlyStopMonitor(max_round=10, higher_better=False, tolerance=1e-4)

for epoch in range(1000):
    print("epoch: ", epoch, end="   ")
    model.train()
    # 分批次训练
    loss1 = []
    for i in range(math.ceil(len(cas_input1) / batch_size)):
        batch = cas_input1[i * batch_size:(i + 1) * batch_size]
        label = train_label[i * batch_size:(i + 1) * batch_size].detach().clone()
        label = label.reshape(-1, 1)
        label = label.to(device)
        batch_sample = all_cas_train[i * batch_size:(i + 1) * batch_size]
        out, loss = model(batch, batch_sample, label)
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
        batch_sample = all_cas_val[i * batch_size:(i + 1) * batch_size]
        label = label2[i * batch_size:(i + 1) * batch_size].detach().clone()
        label = label.reshape(-1, 1)
        label = label.to(device)
        out, loss = model(batch, batch_sample, label)
        loss2.append(loss.item())
    print("val_loss:", np.mean(loss2))

    # 使用早停策略检查是否需要停止训练
    if early_stopping.early_stop_check(np.mean(loss2)):
        print("Early stopping triggered at epoch ", epoch)
        break

    # 如果验证集损失更好，保存模型
    if np.mean(loss2) < best_val_loss:
        best_val_loss = np.mean(loss2)
        torch.save(model.state_dict(), './twitter-model-2d-03.pth')

# 加载保存好的模型
model.load_state_dict(torch.load('twitter-model-2d-03.pth'))
msle_l, male_l, mape_l, pcc_l = [], [], [], []
label3 = torch.tensor(test_label, dtype=torch.float32).to(device).detach().clone()
label3[label3 < 1] = 1
label3 = torch.log2(label3)
loss3 = []
model.eval()
for i in range(math.ceil(len(cas_input3) / batch_size)):
    batch = cas_input3[i * batch_size:(i + 1) * batch_size]
    batch_sample = all_cas_test[i * batch_size:(i + 1) * batch_size]
    label = label3[i * batch_size:(i + 1) * batch_size].detach().clone()
    label = label.reshape(-1, 1)
    label = label.to(device)
    out, loss = model(batch, batch_sample, label)
    loss3.append(loss.item())
    msle_l.append(msle(out.detach().cpu().numpy(), label.detach().cpu().numpy()))
    mape_l.append(mape(out.detach().cpu().numpy(), label.detach().cpu().numpy()))
    male_l.append(male(out.detach().cpu().numpy(), label.detach().cpu().numpy()))
    pcc_l.append(pcc(out.detach().cpu().numpy(), label.detach().cpu().numpy()))
print(
    f"test_loss: {np.mean(loss3)}, msle: {np.mean(msle_l)}, male:{np.mean(male_l)},mape: {np.mean(mape_l)},pcc:{np.mean(pcc_l)}")
