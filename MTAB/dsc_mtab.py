from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from utils import load_data, load_graph
from evaluation import eva
from torch.utils.data import DataLoader, TensorDataset
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    # z is the hidden layer, x_bar is the reconstruction layer
    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h3


class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()
        # AE to obtain internal information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,

            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,

            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # Fill the input "Tensor" with values according to the method
        # and the resulting tensor will have the values sampled from it
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().cuda()

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_h3 = self.ae(x)
        sigma = 0.5
        # GCN Module
        h = self.gnn_1(x, adj)
        # h_>Zl,tra_>H(l)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)
        # The last layer (multiple classification layer with softmax function)
        predict = F.softmax(h, dim=1)
        # Dual Self-supervised Module
        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss
        # qij
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, _mean, _disp, _pi, zinb_loss


# Target distribution
def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_sdcn(dataset, X_raw, sf):
    global p
    model = SDCN(
        n_enc_1=args.n_enc_1,
        n_enc_2=args.n_enc_2,
        n_enc_3=args.n_enc_3,
        n_dec_1=args.n_dec_1,
        n_dec_2=args.n_dec_2,
        n_dec_3=args.n_dec_3,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=1).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    adj = load_graph(args.graph, args.k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_init)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # evaluation.py：ARI\ACC\NMI
    eva(y, y_pred, 0)

    for epoch in range(Para[2]):
        if epoch % 1 == 0:
            _, tmp_q, pred, _, _, _, _, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            # eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, epoch)
            # eva(y, res3, str(epoch) + 'P')

        x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

        binary_crossentropy_loss = F.binary_cross_entropy(q, p)
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()

        zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)
        loss = Balance_para[0] * binary_crossentropy_loss + Balance_para[1] * ce_loss + Balance_para[2] * re_loss + \
               Balance_para[3] * zinb_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # File = [gene_expresion data file, Graph file, Pre-training file, h5 file]
    File = ['mtab', "mtab", 'model/mtab1.pkl', "data/mtab.h5"]
    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1), n_cluster, n_init]
    model_para = [500, 500, 2000]
    # Para = [batch_size, lr, epoch]
    Para = [1024, 1e-4, 300]
    # Cluster_para = [n_cluster, n_z, n_input, n_init]
    Cluster_para = [5, 10, 4889, 20]
    # Balance_para = [binary_crossentropy_loss, ce_loss, re_loss, zinb_loss]
    Balance_para = [0.1, 0.01, 1, 0.1]
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, default=File[0])
    parser.add_argument('--graph', type=str, default=File[1])
    parser.add_argument('--pretrain_path', type=str, default=File[2])
    parser.add_argument('--n_enc_1', default=model_para[0], type=int)
    parser.add_argument('--n_enc_2', default=model_para[1], type=int)
    parser.add_argument('--n_enc_3', default=model_para[2], type=int)
    parser.add_argument('--n_dec_1', default=model_para[2], type=int)
    parser.add_argument('--n_dec_2', default=model_para[1], type=int)
    parser.add_argument('--n_dec_3', default=model_para[0], type=int)

    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--lr', type=float, default=Para[1])

    parser.add_argument('--n_clusters', default=Cluster_para[0], type=int)
    parser.add_argument('--n_z', default=Cluster_para[1], type=int)
    parser.add_argument('--n_input', type=int, default=Cluster_para[2])
    parser.add_argument('--n_init', type=int, default=Cluster_para[3])
    # 解析参数
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    device = torch.device("cuda" if args.cuda else "cpu")
    args.pretrain_path = File[2]
    dataset = load_data(args.name)

    data_mat = h5py.File(File[3], "r+")
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    X = adata.X
    X_raw = adata.raw.X
    sf = adata.obs.size_factors
    dataset1 = TensorDataset(torch.Tensor(X), torch.Tensor(X_raw), torch.Tensor(sf))
    dataloader = DataLoader(dataset1, batch_size=Para[0], shuffle=True)
    train_sdcn(dataset, X_raw, sf)
