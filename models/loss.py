from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def normalize(x, axis=-1):
	"""Normalizing to unit length along the specified dimension.
	Args:
	  x: pytorch Variable
	Returns:
	  x: pytorch Variable, same shape as input
	"""
	x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
	return x

def euclidean_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
	x_normed = F.normalize(x, p=2, dim=1)
	y_normed = F.normalize(y, p=2, dim=1)
	return 1 - torch.mm(x_normed, y_normed.t())

def cosine_similarity(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	"""
	x_normed = F.normalize(x, p=2, dim=1)
	y_normed = F.normalize(y, p=2, dim=1)
	return torch.mm(x_normed, y_normed.t())


def hard_example_mining(dist_mat, labels, xbm_labels):
	"""For each anchor, find the hardest positive and negative sample.
	Args:
	  dist_mat: pytorch Variable, pair wise distance between samples, shape [M, N]
	  labels: pytorch LongTensor, with shape [M]
	  xbm_labels: pytorch LongTensor, with shape [N]
	  return_inds: whether to return the indices. Save time if `False`(?)
	Returns:
	  dist_ap: pytorch Variable, distance(anchor, positive); shape [M]
	  dist_an: pytorch Variable, distance(anchor, negative); shape [M]
	  p_inds: pytorch LongTensor, with shape [N];
		indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
	  n_inds: pytorch LongTensor, with shape [N];
		indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
	NOTE: Only consider the case in which all labels have same num of samples,
	  thus we can cope with all anchors in parallel.
	"""
	assert len(dist_mat.size()) == 2
	M = dist_mat.size(0)
	N = dist_mat.size(1)

	is_pos = labels.expand(N, M).t().eq(xbm_labels.expand(M, N))
	is_neg = labels.expand(N, M).t().ne(xbm_labels.expand(M, N))

	# 难受啊, 到最后还是要用for循环来写
	dist_ap_list = []
	dist_an_list = []
	for i in range(M):
		dist_ap_list.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
		dist_an_list.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
	dist_ap = torch.cat(dist_ap_list)
	dist_an = torch.cat(dist_an_list)
	# for i in range(M):
	# 	dist_ap_i = torch.max(dist_mat[i][is_pos[i]])
	# 	dist_an_i = torch.max(dist_mat[i][is_neg[i]])		
	# 	dist_ap_list.append(dist_ap_i)
	# 	dist_an_list.append(dist_an_i)
	# dist_ap = torch.from_numpy(np.array(dist_ap_list))
	# dist_an = torch.from_numpy(np.array(dist_an_list))

	# dist_ap = dist_ap.squeeze(1)
	# dist_an = dist_an.squeeze(1)

	return dist_ap, dist_an


# ==============
#  Triplet Loss 
# ==============
class TripletHardLoss(object):
	"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
	Loss for Person Re-Identification'."""
	def __init__(self, margin=None, metric="euclidean"):
		self.margin = margin
		self.metric = metric
		if margin is not None:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def __call__(self, global_feat, labels, normalize_feature=False):
		if normalize_feature:
			global_feat = normalize(global_feat, axis=-1)

		if self.metric == "euclidean":
			dist_mat = euclidean_dist(global_feat, global_feat)
		elif self.metric == "cosine":
			dist_mat = cosine_dist(global_feat, global_feat)
		else:
			raise NameError

		dist_ap, dist_an = hard_example_mining(
			dist_mat, labels)
		y = dist_an.new().resize_as_(dist_an).fill_(1)

		if self.margin is not None:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss


class TripletHardLoss_xbm(object):
	"""Modified from 44's open-reid (https://github.com/Cysu/open-reid).
	Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
	Loss for Person Re-Identification'."""
	def __init__(self, margin=None, metric="euclidean"):
		self.margin = margin
		self.metric = metric
		if margin is not None:
			self.ranking_loss = nn.MarginRankingLoss(margin=margin)
		else:
			self.ranking_loss = nn.SoftMarginLoss()

	def __call__(self, global_feat, labels, xbm_feat, xbm_labels, normalize_feature=False):
		if normalize_feature:
			global_feat = normalize(global_feat, axis=-1)

		if self.metric == "euclidean":
			dist_mat = euclidean_dist(global_feat, xbm_feat)
		elif self.metric == "cosine":
			dist_mat = cosine_dist(global_feat, xbm_feat)
		else:
			raise NameError

		dist_ap, dist_an = hard_example_mining_xbm(
			dist_mat, labels, xbm_labels) 
		y = dist_an.new().resize_as_(dist_an).fill_(1)

		if self.margin is not None:
			loss = self.ranking_loss(dist_an, dist_ap, y)
		else:
			loss = self.ranking_loss(dist_an - dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss


dist = cosine_dist(flat_latents, self.embedding.weight)

encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
# Convert to one-hot encodings
device = latents.device
encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

# Quantize the latents
quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

dist_ap, dist_an = hard_example_mining(dist)
y = dist_an.new().resize_as_(dist_an).fill_(1)
loss = self.ranking_loss(dist_an, dist_ap, y)

