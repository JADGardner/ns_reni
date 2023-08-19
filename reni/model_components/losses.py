# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Collection of RENI Losses.
"""
import torch
from torch import nn

class KLD(nn.Module):
    """
    Kullback-Leibler Divergence (KLD) loss, normalised by the number of latent dimensions.
    """

    def __init__(self, Z_dims=1):
        super(KLD, self).__init__()
        self.Z_dims = Z_dims

    def forward(self, mu, log_var):
        """
        forward method for the KLD class.

        Parameters:
        mu (torch.Tensor): The mean.
        log_var (torch.Tensor): The logarithm of the variance.

        Returns:
        torch.Tensor: The Kullback-Leibler divergence.
        """
        kld = -0.5 * ((1 + log_var - mu.pow(2) - log_var.exp()).view(mu.shape[0], -1)).sum(1)
        kld /= self.Z_dims
        kld = kld.sum(0)
        return kld

class ScaleInvariantLogLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantLogLoss, self).__init__()

    def forward(self, log_predicted, log_gt):
        R = log_predicted - log_gt

        term1 = torch.mean(R**2)
        term2 = torch.pow(torch.sum(R), 2) / (log_predicted.numel()**2)

        loss = term1 - term2

        return loss

class ScaleInvariantGradientMatchingLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantGradientMatchingLoss, self).__init__()

    def forward(self, log_predicted, log_predicted_rolled, log_gt, log_gt_rolled):
        """"
        log_predicted: predicted rgb in log domain : (N, 3)
        log_predicted_rolled: predicted rgb in log domain sampled with directions rolled by 1 (N, 3)
        log_gt: ground truth rgb in log domain : (N, 3)
        log_gt_rolled: ground truth rgb in log domain sampled with directions rolled by 1 (N, 3)
        """
        #  l1 penalty on differences in log-depth gradients between the predicted and ground truth depth map

        # estimate gradients using finite differences
        grad_log_gt = torch.abs(log_gt_rolled - log_gt)
        grad_log_predicted = torch.abs(log_predicted_rolled - log_predicted)

        loss = torch.mean(torch.abs(grad_log_gt - grad_log_predicted))

        return loss

