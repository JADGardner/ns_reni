import torch
import torch.nn as nn
from vnn.vn_layers import *


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

class VN_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, k):
        super(VN_MLP, self).__init__()
        self.k = k
        self.in_dim = in_dim

        net = []
        
        for i in hidden_dims:
            net.append(VNLinear(in_channels=in_dim, out_channels=i))
            net.append(VNBatchNorm(num_features=i, dim=5))
            net.append(VNLeakyReLU(in_channels=i, share_nonlinearity=False, negative_slope=0.0))
            in_dim = i

        net.append(VNLinear(in_channels=in_dim, out_channels=out_dim))
        net.append(VNBatchNorm(num_features=out_dim, dim=5))
        net.append(VNLeakyReLU(in_channels=out_dim, share_nonlinearity=False, negative_slope=0.0))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        p = x.unsqueeze(1).transpose(2, 3) # [B, N, 3] -> [B, 1, 3, N]
        feat = get_graph_feature_cross(p, k=self.k, dims=3)
        x = self.net(feat)
        x = x.transpose(2, 3).squeeze() # [B, C, 3, N, 1] -> [B, C, N, 3]
        return x


class VN_DGCNN(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.dim=dim
        
        self.conv_pos = VNLinearLeakyReLU(3, 64, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(64, 2*hidden_dim)
        self.fc_0 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_1 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_2 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_3 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)
        
        
        self.actvn_0 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_2 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_3 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=4, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=4, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        
    def forward(self, p):
        batch_size = p.size(0)
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)
        
        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)
        
        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''

        p = p.unsqueeze(1).transpose(2, 3) # [B, N, 3] -> [B, 1, 3, N]
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k, dims=3) # [B, 1, 3, N] -> [B, 3, 3, N, K]
        net = self.conv_pos(feat) # [B, 3, 3, N, K] -> [B, Z, 3, N, K]
        net = self.pool(net, dim=-1) # [B, Z, 3, N, K] -> [B, Z, 3, N]

        net = self.fc_pos(net) # [B, Z, 3, N] -> [B, Z↑, 3, N] Where Z↑ is the output dimension of the layer

        net = self.fc_0(self.actvn_0(net)) # [B, Z, 3, N] -> [B, Z↑, 3, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size()) # [B, Z, 3, N]
        net = torch.cat([net, pooled], dim=1) # [B, 2Z, 3, N]

        net = self.fc_1(self.actvn_1(net)) # [B, Z, 3, N] -> [B, Z↑, 3, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size()) # [B, Z, 3, N]
        net = torch.cat([net, pooled], dim=1) # [B, 2Z, 3, N]

        net = self.fc_2(self.actvn_2(net)) # [B, Z, 3, N] -> [B, Z↑, 3, N]
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size()) # [B, Z, 3, N]
        net = torch.cat([net, pooled], dim=1) # [B, 2Z, 3, N]

        net = self.fc_3(self.actvn_3(net)) # [B, Z, 3, N] -> [B, Z↑, 3, N]

        # I commented this as was pooling over the 'number of points' N dimension
        # net = model.pool(net, dim=-1) # [B, Z↑, 3, N] -> [B, Z↑, 3]

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            c = c.permute(0, 1, 3, 2)
            c_std = c_std.permute(0, 1, 3, 2)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std

        return c