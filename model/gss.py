import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GSS(nn.Module):
    def __init__(self, x_dim, num_layer=2, normalize=True):
        super(GSS, self).__init__()

        self.x_dim = x_dim
        self.h = nn.Linear(x_dim, x_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.normalize = normalize
        self.num_layer = num_layer

    def forward(self, x, adj):
        for i in range(self.num_layer):
            if self.normalize:
                x = x / (torch.sum(x,dim=1) + 0.0001).view(-1, 1)
            pre = self.h(torch.spmm(adj, x))
            x = F.elu(pre)
            if self.normalize:
                x = self.dropout(x)
        emb = F.normalize(x,p=2,dim=1)
        pred = torch.mm(emb,emb.T())
        out = F.relu(pred)
        return out

class GSS_loss():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def gss_loss(self, logits):
        losses = -0.5 * self.alpha * (logits - self.beta) ** 2
        return torch.mean(losses)


# import scipy.sparse as sps
# import numpy as np
# def preprocess_graph(adj):
#     adj_ = adj + sp.eye(adj.shape[0])
#     rowsum = np.array(adj_.sum(1))
#     degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
#     adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).transpose()
#     return sp.csr_matrix(adj_normalized)
# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return torch.sparse.FloatTensor(indices, coo.data.astype(np.float32), coo.shape).to_dense()
# x_sim = np.matmul(X.T, X)
# k = 5
# x_sim_top = np.argpartition(x_sim, -k, 1)[:, -k:]
# x_adj = np.zeros(x_sim.shape)
# for i in range(x_adj.shape[0]):
#     x_adj[i, x_sim_top[i]] = x_sim[i, x_sim_top[i]]
#     x_adj[x_sim_top[i], i] = x_sim[i, x_sim_top[i]]
#     x_adj[i, i] = 0
# x_adj = sp.csr_matrix(x_adj)
# x_adj_normed = preprocess_graph(x_adj)
# x_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(x_adj_normed)