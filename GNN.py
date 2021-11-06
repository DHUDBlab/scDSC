import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import initializers
from torch.autograd import Variable

# class GNNLayer(tf.Module):
#     def __init__(self, in_features, out_features):
#         super(GNNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         shape=(in_features, out_features)
#         initializer = tf.initializers.GlorotUniform()
#         self.weight = tf.Variable(initializer(shape=shape))
#
#     def forward(self, features, adj, active=True):
#         support = tf.matmul(features, self.weight)
#         output = tf.sparse.sparse_dense_matmul(adj, support)
#         if active:
#             output = tf.nn.relu(output)
#         return output

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # When passing through the network layer, the input and output variances are the same
        # including forward propagation and backward propagation

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output
