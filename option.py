import argparse

parser = argparse.ArgumentParser('hyper parameters of casflow-pytorch')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_seq', type=int, default=50, help='Max length of cascade sequence.')
parser.add_argument('--emb_dim', type=int, default=80, help='Embedding dimension (cascade emb_dim + global emb_dim')
parser.add_argument('--z_dim', type=int, default=64, help='Dimension of latent variable z.')
parser.add_argument('--rnn_units', type=int, default=128, help='Number of RNN units.')
parser.add_argument('--n_flows', type=int, default=8, help='Number of NF transformations.')
parser.add_argument('--verbose', type=int, default=2, help='Verbose.')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')

parser.add_argument('--dir_data', type=str, default='./dataset')
parser.add_argument('--data_set', type=str, default='twitter', choices=('aps', 'weibo', 'twitter'))
parser.add_argument('--bipartite', action='store_true', default=True)
parser.add_argument('--mode', type=str, default='sad', choices=('origin', 'gdn', 'sad'))  # 模型名
parser.add_argument('--add_scl', action='store_true', default=False)
parser.add_argument('--module_type', type=str, default='graph_sum', choices=('graph_attention', 'graph_sum'))
parser.add_argument('--mask_label', action='store_true', default=False)
parser.add_argument('--mask_ratio', type=float, default=0.5)

##data param
parser.add_argument('--n_neighbors', type=int, default=20, help='Maximum number of connected edge per node')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--num_data_workers', type=int, default=25)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--accelerator', type=str, default='ddp')

##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--drop_out', type=float, default=0.2)
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=5e-4)

# node_dim edge_dim
parser.add_argument('--node_dim', type=int, default=40)
parser.add_argument('--edge_dim', type=int, default=40)
parser.add_argument('--hidden_dim', type=int, default=40)
parser.add_argument('--input_dim', type=int, default=40)

args = parser.parse_args()
