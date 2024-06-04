from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


#patchcore
from typing import Tuple, Union, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import ListConfig

from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d
from torch import Tensor, nn

import numpy as np
from sklearn.utils.random import sample_without_replacement

"""Gaussian blurring via pytorch."""

class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d.

    Makes use of kornia functions, but most notably the kernel is not computed
    during the forward pass, and does not depend on the input size. As a caveat,
    the number of channels that are expected have to be provided during initialization.
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        sigma: Union[Tuple[float, float], float],
        channels: int,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        """Initialize model, setup kernel etc..

        Args:
            kernel_size (Union[Tuple[int, int], int]): size of the Gaussian kernel to use.
            sigma (Union[Tuple[float, float], float]): standard deviation to use for constructing the Gaussian kernel.
            channels (int): channels of the input
            normalize (bool, optional): Whether to normalize the kernel or not (i.e. all elements sum to 1).
                Defaults to True.
            border_type (str, optional): Border type to use for padding of the input. Defaults to "reflect".
            padding (str, optional): Type of padding to apply. Defaults to "same".
        """
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.kernel: Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)
        self.channels = channels
        self.kernel.unsqueeze_(0).unsqueeze_(0)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Blur the input with the computed Gaussian.

        Args:
            input_tensor (Tensor): Input tensor to be blurred.

        Returns:
            Tensor: Blurred output tensor.
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        #import ipdb
        #ipdb.set_trace()
        output = F.conv2d(input_tensor, self.kernel.cuda(), groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""
    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(self, patch_scores: torch.Tensor, feature_map_shape: torch.Size) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            feature_map_shape (torch.Size): 2-D feature map shape (width, height)

        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        width, height = feature_map_shape
        batch_size = len(patch_scores) // (width * height)

        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, width, height))
        anomaly_map = F.interpolate(anomaly_map, size=(self.input_size[0], self.input_size[1]))
        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: torch.Tensor) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
        Returns:
            torch.Tensor: Image-level anomaly scores
        """
        max_scores = torch.argmax(patch_scores[:, 0])
        confidence = torch.index_select(patch_scores, 0, max_scores)
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * torch.max(patch_scores[:, 0])
        return score

    def forward(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly
        Expects `feature_map_shape` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array, feature_map_shape=feature_map_shape)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        if "feature_map_shape" not in kwargs:
            raise ValueError(f"Expected key `feature_map_shape`. Found {kwargs.keys()}")

        patch_scores = kwargs["patch_scores"]
        feature_map_shape = kwargs["feature_map_shape"]

        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)
        #anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, None

"""This module comprises PatchCore Sampling Methods for the embedding.

- k Center Greedy Method
    Returns points that minimizes the maximum distance of any point to a center.
    . https://arxiv.org/abs/1708.00489
"""

class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        embedding (Tensor): Embedding vector extracted from a CNN
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: Tensor
        self.min_distances: Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = None

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        for _ in range(self.coreset_size):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset


"""This module comprises PatchCore Sampling Methods for the embedding.

- Random Sparse Projector
    Sparse Random Projection using PyTorch Operations
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class NotFittedError(ValueError, AttributeError):
    """Raise Exception if estimator is used before fitting."""


class SparseRandomProjection:
    """Sparse Random Projection using PyTorch operations.

    Args:
        eps (float, optional): Minimum distortion rate parameter for calculating
            Johnson-Lindenstrauss minimum dimensions. Defaults to 0.1.
        random_state (Optional[int], optional): Uses the seed to set the random
            state for sample_without_replacement function. Defaults to None.
    """

    def __init__(self, eps: float = 0.1, random_state: Optional[int] = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int):
        """Random sparse matrix. Based on https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf.

        Args:
            n_features (int): Dimentionality of the original source space

        Returns:
            Tensor: Sparse matrix of shape (n_components, n_features).
                The generated Gaussian random matrix is in CSR (compressed sparse row)
                format.
        """

        # Density 'auto'. Factorize density
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components

        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = torch.zeros((self.n_components, n_features), dtype=torch.float64)
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features, n_samples=nnz_idx, random_state=self.random_state
                    ),
                    dtype=torch.int64,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data.double()

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    def johnson_lindenstrauss_min_dim(self, n_samples: int, eps: float = 0.1):
        """Find a 'safe' number of components to randomly project to.

        Ref eqn 2.1 https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf

        Args:
            n_samples (int): Number of samples used to compute safe components
            eps (float, optional): Minimum distortion rate. Defaults to 0.1.
        """

        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: Tensor) -> "SparseRandomProjection":
        """Generates sparse matrix from the embedding tensor.

        Args:
            embedding (Tensor): embedding tensor for generating embedding

        Returns:
            (SparseRandomProjection): Return self to be used as
            >>> generator = SparseRandomProjection()
            >>> generator = generator.fit()
        """
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self.johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)

        return self

    def transform(self, embedding: Tensor) -> Tensor:
        """Project the data by using matrix product with the random matrix.

        Args:
            embedding (Tensor): Embedding of shape (n_samples, n_features)
                The input data to project into a smaller dimensional space

        Returns:
            projected_embedding (Tensor): Sparse matrix of shape
                (n_samples, n_components) Projected array.
        """
        if self.sparse_random_matrix is None:
            raise NotFittedError("`fit()` has not been called on SparseRandomProjection yet.")

        projected_embedding = embedding @ self.sparse_random_matrix.T.float()
        return projected_embedding


#Patchcore2
import abc
from sklearn.metrics import pairwise_distances

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None

class kCenterGreedy2(SamplingMethod):

  def __init__(self, X, y, seed, metric='euclidean'):
    self.X = X
    self.y = y
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, model, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      self.features = model.transform(self.X)
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for _ in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))


    self.already_selected = already_selected

    return new_batch