
# TODO(lujiang): put it into the third-party
# MIT License

# Copyright (c) 2018 Nam Vo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Metric learning functions.

Codes are modified from:
https://github.com/lugiavn/generalization-dml/blob/master/nams.py
"""

import numpy as np
import torch
import torchvision


def pairwise_distances(x, y=None):
  """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  return torch.clamp(dist, 0.0, np.inf)

# Updated version of MyTripletLossFunc using the new static method style.
class MyTripletLossFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, features, triplets):
        # Save tensors for backward pass
        ctx.save_for_backward(features)
        ctx.triplets = triplets

        # Compute pairwise distances
        distances = pairwise_distances(features).cpu().numpy()

        # Compute the loss
        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in triplets:
            w = 1.0
            triplet_count += w
            loss += w * np.log(1 + np.exp(distances[i, j] - distances[i, k]))
            if distances[i, j] < distances[i, k]:
                correct_count += 1

        loss /= triplet_count
        return torch.FloatTensor((loss,))

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        features, = ctx.saved_tensors
        triplets = ctx.triplets
        features_np = features.cpu().numpy()
        
        # Initialize gradients
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()

        # Compute gradients
        for i, j, k in triplets:
            w = 1.0
            f = 1.0 - 1.0 / (1.0 + np.exp(ctx.distances[i, j] - ctx.distances[i, k]))
            grad_features_np[i, :] += w * f * (features_np[i, :] - features_np[j, :]) / len(triplets)
            grad_features_np[j, :] += w * f * (features_np[j, :] - features_np[i, :]) / len(triplets)
            grad_features_np[i, :] += -w * f * (features_np[i, :] - features_np[k, :]) / len(triplets)
            grad_features_np[k, :] += -w * f * (features_np[k, :] - features_np[i, :]) / len(triplets)

        # Convert gradients back to tensor
        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])

        # Scale by the gradient from the output
        grad_features *= float(grad_output.data[0])
        return grad_features, None  # None for triplets, as it's not a tensor

# TripletLoss wrapper for using the custom loss in a model
class TripletLoss(torch.nn.Module):
    """Class for the triplet loss."""
    def __init__(self, pre_layer=None):
        super(TripletLoss, self).__init__()
        self.pre_layer = pre_layer

    def forward(self, x, triplets):
        if self.pre_layer is not None:
            x = self.pre_layer(x)
        loss = MyTripletLossFunc.apply(x, triplets)
        return loss

class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features
