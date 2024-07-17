import pickle

import numpy as np
from absl import app, flags

from graphwave.graphwave import graphwave_alg
from sparse_matrix_factorization import *
import torch

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('cg_emb_dim', 40, 'Cascade graph embedding dimension.')
flags.DEFINE_integer('gg_emb_dim', 40, 'Global graph embedding dimension.')
flags.DEFINE_integer('max_seq', 25, 'Max length of cascade sequence.')
flags.DEFINE_integer('num_s', 2, 'Number of s for spectral graph wavelets.')

# twitter 1day
# flags.DEFINE_integer('observation_time', 1 * 24 * 60 * 60, 'Observation time.')

# weibo
flags.DEFINE_integer('observation_time', 3600, 'Observation time.')

# aps 3y
# flags.DEFINE_integer('observation_time', 1095, 'Observation time.')

# paths
flags.DEFINE_string('input', './dataset/aps/less25/5y/', 'Dataset path.')
flags.DEFINE_string('gg_path', 'global_graph_less25.pkl', 'Global graph path.')


def sequence2list(filename):
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            paths = line.strip().split('\t')[:-1][:FLAGS.max_seq + 1]
            graphs[paths[0]] = list()
            for i in range(1, len(paths)):
                nodes = paths[i].split(':')[0]
                time = paths[i].split(':')[1]
                graphs[paths[0]].append([[int(x) for x in nodes.split(',')], int(time)])

    return graphs


def read_labels(filename):
    labels = dict()
    with open(filename, 'r') as f:
        for line in f:
            id = line.strip().split('\t')[0]
            labels[id] = line.strip().split('\t')[-1]

    return labels


def write_cascade(graphs, labels, id2row, filename, gg_emb, weight=True):
    """
    Input: cascade graphs, global embeddings
    Output: cascade embeddings, with global embeddings appended
    """
    y_data = list()
    cascade_input = list()
    global_input = list()
    cascade_i = 0
    cascade_size = len(graphs)
    total_time = 0

    src_l, dst_l, e_idx_l, ts_l = [], [], [], []
    edge_data = []
    # for each cascade graph, generate its embeddings via wavelets
    for key, graph in graphs.items():
        start_time = time.time()
        y = int(labels[key])

        # lists for saving embeddings
        cascade_temp = list()
        global_temp = list()

        # build graph
        g = nx.Graph()
        nodes_index = list()
        nodes_index1 = list()
        list_edge = list()
        cascade_embedding = list()
        global_embedding = list()
        times = list()
        t_o = FLAGS.observation_time
        # add edges into graph
        for path in graph:
            t = path[1]

            if t >= t_o:
                continue
            nodes = path[0]
            if len(nodes) == 1:
                nodes_index.extend(nodes)
                # nodes_index1.extend(nodes)
                times.append(1)
                src_l.append((nodes[0]))
                dst_l.append((nodes[0]))
                e_idx_l.append(int(key))
                ts_l.append(t)
                continue
            else:
                e_idx_l.append(int(key))
                ts_l.append(t)
                src_l.append((nodes[0]))
                dst_l.append((nodes[-1]))
                nodes_index.extend([nodes[-1]])
                # nodes_index.extend([nodes[0]])
                # nodes_index1.extend(nodes[-1])
                # nodes_index1.extend(nodes[0])
            if weight:
                edge = (nodes[-1], nodes[-2], (1 - t / t_o))  # weighted edge
                times.append(1 - t / t_o)
            else:
                edge = (nodes[-1], nodes[-2])
            list_edge.append(edge)

        if weight:
            g.add_weighted_edges_from(list_edge)
        else:
            g.add_edges_from(list_edge)

        # this list is used to make sure the node order of `chi` is same to node order of `cascade`
        nodes_index_unique = list(set(nodes_index))
        nodes_index_unique.sort(key=nodes_index.index)

        nodes_index_unique1 = list(set(g.nodes))
        # nodes_index_unique1.sort(key=nodes_index1.index)
        # embedding dim check
        d = FLAGS.cg_emb_dim / (2 * FLAGS.num_s)
        if FLAGS.cg_emb_dim % 4 != 0:
            raise ValueError

        # create edge_index
        # 把node_index_unique中的节点映射成id
        id2node = {}
        for i in range(len(nodes_index_unique1)):
            id2node[nodes_index_unique1[i]] = i
        edge_index = []
        # 遍历g.edges
        for edge in g.edges:
            # 把每个值替换成字典的键对应的值
            edge_index.append([id2node[edge[0]], id2node[edge[1]]])
        edge_index = np.array(edge_index).T
        # 把edge_index转成tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # generate cascade embeddings
        chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                  taus='auto', verbose=False,
                                  nodes_index=nodes_index_unique,
                                  nb_filters=FLAGS.num_s)

        # save embeddings into list
        for node in nodes_index:
            cascade_embedding.append(chi[nodes_index_unique.index(node)])
            global_embedding.append(gg_emb[id2row[node]])

        # concat node features to node embedding
        if weight:
            cascade_embedding = np.concatenate([np.reshape(times, (-1, 1)),
                                                np.array(cascade_embedding)[:, 1:]],
                                               axis=1)

        # save embeddings
        cascade_temp.extend(cascade_embedding)
        global_temp.extend(global_embedding)
        cascade_input.append(cascade_temp)
        global_input.append(global_temp)

        # save labels
        y_data.append(y)

        # save edges
        edge_data.append(edge_index)

        # log
        total_time += time.time() - start_time
        cascade_i += 1
        if cascade_i % 1000 == 0:
            speed = total_time / cascade_i
            eta = (cascade_size - cascade_i) * speed
            print('{}/{}, eta: {:.2f} mins'.format(
                cascade_i, cascade_size, eta / 60))

    # write concatenated embeddings into file
    with open(filename, 'wb') as f:
        pickle.dump((cascade_input, global_input, y_data), f)

    # 把edge_data写入文件
    with open(filename[:-4] + '_edge.pkl', 'wb') as f:
        pickle.dump(edge_data, f)


def write_ml(graphs, filename):
    """
    Input: cascade graphs
    Output: (u, v, e_idx, ts) for each edge
    """
    cas_list = []
    # for each cascade graph, generate its embeddings via wavelets
    for key, graph in graphs.items():
        src_l, dst_l, e_idx_l, ts_l = [], [], [], []
        cas = {}
        t_o = FLAGS.observation_time
        # add edges into graph
        for path in graph:
            t = path[1]

            if t >= t_o:
                continue
            nodes = path[0]
            if len(nodes) == 1:
                src_l.append((nodes[0]))
                dst_l.append((nodes[0]))
                e_idx_l.append(int(key))
                ts_l.append(t)
                continue
            else:
                e_idx_l.append(int(key))
                ts_l.append(t)
                src_l.append((nodes[-2]))
                dst_l.append((nodes[-1]))
        cas["src_l"] = src_l
        cas["dst_l"] = dst_l
        cas["e_idx_l"] = e_idx_l
        cas["ts_l"] = ts_l
        cas_list.append(cas)
    with open(FLAGS.input + filename + '_ml.pkl', 'wb') as f:
        pickle.dump(cas_list, f)
    return cas_list


def main(argv):
    time_start = time.time()

    # get the information of nodes/users of cascades
    graph_train = sequence2list(FLAGS.input + 'train_less25.txt')
    graph_val = sequence2list(FLAGS.input + 'val_less25.txt')
    graph_test = sequence2list(FLAGS.input + 'test_less25.txt')

    # get the information of labels of cascades
    label_train = read_labels(FLAGS.input + 'train_less25.txt')
    label_val = read_labels(FLAGS.input + 'val_less25.txt')
    label_test = read_labels(FLAGS.input + 'test_less25.txt')

    # load global graph and generate id2row
    with open(FLAGS.input + FLAGS.gg_path, 'rb') as f:
        gg = pickle.load(f)

    # sparse matrix factorization
    print('Generating embeddings of nodes in global graph.')
    model = SparseMatrixFactorization(gg, FLAGS.gg_emb_dim)
    gg_emb = model.pre_factorization(model.matrix, model.matrix)

    ids = [int(xovee) for xovee in gg.nodes()]
    id2row = dict()
    i = 0
    for id in ids:
        id2row[id] = i
        i += 1

    print('Start writing train set into file.')
    write_cascade(graph_train, label_train, id2row, FLAGS.input + 'train_less25.pkl', gg_emb)
    write_ml(graph_train, "train_less25")
    print('Start writing val set into file.')
    write_ml(graph_val, "val_less25")
    write_cascade(graph_val, label_val, id2row, FLAGS.input + 'val_less25.pkl', gg_emb)
    print('Start writing test set into file.')
    write_ml(graph_test, "test_less25")
    write_cascade(graph_test, label_test, id2row, FLAGS.input + 'test_less25.pkl', gg_emb)

    print('Processing time: {:.2f}s'.format(time.time() - time_start))


if __name__ == '__main__':
    app.run(main)
