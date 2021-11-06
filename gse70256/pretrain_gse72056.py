import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
import torch.nn.functional as F



class AE(nn.Module):
    # 定义模型框架，初始化
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z1, n_z2, n_z3):
        super(AE, self).__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)


    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar,  z3


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=Para[0], shuffle=True)
    print(model)
    # Adam
    optimizer = Adam(model.parameters(), lr=Para[1])
    for epoch in range(Para[2]):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            x_bar, _ = model(x)

            x_bar = x_bar.cpu()
            x = x.cpu()
            loss = F.mse_loss(x_bar, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z3 = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=Cluster_para[0], n_init=Cluster_para[1]).fit(z3.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)
        # Generate a pre-trained model
        torch.save(model.state_dict(), File[0])

# File = [Pre-training file, gene_expresion data file, labels file]
File = ['model/gse70256_.pkl', 'data/gse70256_filter.txt', 'data/gse70256_filter_label.txt']
# Para = [batch_size, lr, epoch]
Para = [1024, 1e-4, 100]
# model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
model_para = [1000, 1000, 4000]
# Cluster_para = [n_cluster, n_init, n_input, n_z1, n_z2, n_z3 ]
Cluster_para = [7, 20, 3276, 2000, 500, 10]

model = AE(
            n_enc_1=model_para[0], n_enc_2=model_para[1], n_enc_3=model_para[2],
            n_dec_1=model_para[2], n_dec_2=model_para[1], n_dec_3=model_para[0],
            n_input=Cluster_para[2], n_z1=Cluster_para[3], n_z2=Cluster_para[4], n_z3=Cluster_para[5], ).cuda()


x = np.loadtxt(File[1], dtype=float)
y = np.loadtxt(File[2], dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)
