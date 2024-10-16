from typing import Optional, Tuple
import numpy as np
import numba as nb
from motpy.distributions.gaussian import Gaussian, merge_gaussians
# Import matrix inverse from scipy
import scipy


def likelihood(x, mu, P):
  scale = 1 / np.sqrt((2 * np.pi * np.linalg.det(P)))
  y = x - mu
  exp = np.exp(-0.5 * np.einsum('...i,...ii,...i->...',
               y, np.linalg.inv(P), y))
  return scale * exp


# def merge(w: np.ndarray,
#           mu: np.ndarray,
#           P: np.ndarray,
#           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#   w_merged = np.sum(w, axis=-1, keepdims=True)
#   mu_merged = np.einsum('...i, ...ij -> ...j', w, mu) / w_merged

#   y = mu - mu_merged
#   y_outer = np.einsum('...i, ...j -> ...ij', y, y)
#   P_merged = np.einsum('...i, ...ijk -> ...jk', w / w_merged, P + y_outer)
#   return w_merged, mu_merged, P_merged


def runnalls_reduce(distribution: Gaussian,
                    max_n: int,
                    ) -> Gaussian:
  """
  Runnalls mixture reduction algorithm as described in [1].

  Unlike the algorithm in the paper, we use the trace when computing the costs instead of the determinant. This is much faster and gives similar results if the covariance components have similar magnitudes in all dimensions. If they don't, the trace can be scaled by user-specified values.

  [1] Crouse2011 - A Look at Gaussian Mixture Reduction Algorithms


  Parameters
  ----------
  distribution : Gaussian
      Input distribution
  max_n : int
      Desired number of components in the reduced distribution

  Returns
  -------
  Gaussian
      The reduced distribution
  """
  valid = np.ones(distribution.size, dtype=bool)
  inds = np.arange(distribution.size)
  w = distribution.weight.copy()
  mu = distribution.mean.copy()
  P = distribution.covar.copy()

  while np.count_nonzero(valid) > max_n:
    # Compute the cost for all pairs of components
    wi = np.expand_dims(w[valid], axis=-1)
    wj = np.expand_dims(w[valid], axis=-2)

    mu_i = np.expand_dims(mu[valid], axis=-2)
    mu_j = np.expand_dims(mu[valid], axis=-3)
    Pi = np.expand_dims(P[valid], axis=-3)
    Pj = np.expand_dims(P[valid], axis=-4)

    wij = wi + wj
    mu_ij = (wi[..., None]*mu_i + wj[..., None]*mu_j) / wij[..., None]
    yi = mu_i - mu_ij
    yj = mu_j - mu_ij
    Pij = 1 / wij[..., None, None] * (
        wi[..., None, None] * (Pi + yi[..., :, None] * yi[..., None, :]) +
        wj[..., None, None] * (Pj + yj[..., :, None] * yj[..., None, :]))
    # TODO: May be even easier to scale the trace automatically by studentizing the covariance matrices or scaling them to the range [0, 1]
    # Compute the max of Pij for each matrix value
    trace_ij = np.trace(Pij,
                        axis1=-1, axis2=-2)
    trace_i = np.trace(Pi,
                       axis1=-1, axis2=-2)
    trace_j = np.trace(Pj,
                       axis1=-1, axis2=-2)
    c = (0.5 * wij * trace_ij - (wi * trace_i) - (0.5 * wj * trace_j))

    # Merge the two components with the smallest cost
    # Set the diagonal to infinity so that we don't merge a component with itself
    c[np.diag_indices_from(c)] = np.inf
    i, j = np.unravel_index(np.argmin(c), c.shape)
    valid_inds = inds[valid]
    w[valid_inds[i]] = wij[i, j]
    mu[valid_inds[i]] = mu_ij[i, j]
    P[valid_inds[i]] = Pij[i, j]
    valid[valid_inds[j]] = False

  distribution = Gaussian(
      mean=mu[valid],
      covar=P[valid],
      weight=w[valid]
  )
  return distribution


def west_reduce(distribution: Gaussian,
                max_n: int,
                gamma: float = np.inf) -> Gaussian:
  mu = distribution.mean.copy()
  P = distribution.covar.copy()
  w = distribution.weight.copy()
  valid = np.ones(distribution.size, dtype=bool)

  while np.count_nonzero(valid) > max_n:
    # Compute modified weights for each component in the current mixture
    traces = np.trace(P[valid], axis1=-1, axis2=-2)
    wmod = w[valid] / traces

    # Choose the i with the smallest modified weight
    ic = np.argmin(wmod)
    i = np.where(valid)[0][ic]

    mu_i = mu[i][None, ...]
    mu_j = mu[valid]
    P_i = P[i, None, :, :]
    P_j = P[valid]
    P_ipj = P_i + P_j

    # Compute ISE cost between each pair of components
    l1 = likelihood(mu_j, mu_i, P_ipj)
    l2 = likelihood(mu_i, mu_i, 2 * P_i) + likelihood(mu_j, mu_j, 2 * P_j)
    c = -2 * l1 + l2
    c[ic] = np.inf

    # Choose the j with the smallest ISE cost and merge
    jc = np.argmin(c)
    j = np.where(valid)[0][jc]
    if c[jc] > gamma:
      break
    w[i], mu[i], P[i] = merge_gaussians(
        w=w[[i, j]],
        mu=mu[[i, j]],
        P=P[[i, j]]
    )

    valid[j] = False

  w_merged = w[valid]
  mu_merged = mu[valid]
  P_merged = P[valid]
  return Gaussian(weight=w_merged, mean=mu_merged, covar=P_merged)


def mean_distance_reduce(distribution: Gaussian, max_n: int):
    # Merge the Gaussian distribution according to the following rules:
    # - Find the max_n largest-weight components
    # - For the rest of the components, find the closest component of the max_n largest
    # - Merge the components with their assigned groups (more than one component may be assigned to the same larger component)

  n = distribution.size
  if n <= max_n:
    return distribution
  w = distribution.weight.copy()
  mu = distribution.mean.copy()
  P = distribution.covar.copy()

  wmod = w / np.trace(P, axis1=-1, axis2=-2)
  top_inds = np.argpartition(wmod, -max_n)[-max_n:]
  bottom_inds = np.setdiff1d(np.arange(n), top_inds)

  # Nearest neighbor assignment
  distances = np.sum(
      (mu[top_inds, None, :] - mu[None, bottom_inds, :])**2, axis=-1)
  closest = np.zeros(n, dtype=int)
  closest[:max_n] = np.arange(max_n)
  closest[max_n:] = np.argmin(distances, axis=0)

  # Get group indices for each component to keep
  group_sizes = np.bincount(closest)
  largest_group = group_sizes.max()
  group_inds = np.array([
      np.pad(
          np.where(closest == i)[0],
          pad_width=(0, largest_group - group_sizes[i]),
          mode='constant',
          constant_values=-1
      ) for i in range(max_n)
  ])

  # Merge groups
  w_group = np.where(group_inds != -1, w[group_inds], 0)
  mu_group = np.where(group_inds[..., None] != -1, mu[group_inds], 0)
  P_group = np.where(group_inds[..., None, None] != -1, P[group_inds], 0)

  w_merged, mu_merged, P_merged = merge_gaussians(
      weights=w_group, means=mu_group, covars=P_group)

  return Gaussian(weight=w_merged, mean=mu_merged, covar=P_merged)


if __name__ == '__main__':
  # w = np.array([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.1, 0.08, 0.06])
  # mu = np.array([1.45, 2.20, 0.67, 0.48, 1.49, 0.91,
  #               1.01, 1.42, 2.77, 0.89])[..., None]
  # P = np.array([0.0487, 0.0305, 0.1171, 0.0174, 0.0295,
  #               0.0102, 0.0323, 0.0380, 0.0115, 0.0679])[..., None, None]
  N = 100
  w = np.ones(N)
  mu = np.random.rand(N, 2)
  P = np.eye(2)[None, ...].repeat(N, axis=0)

  distribution = Gaussian(
      weight=w,
      mean=mu,
      covar=P,
  )
  max_n = 64
  # reduced = runnalls_reduce(distribution, max_n)
  reduced = mean_distance_reduce(distribution, max_n)
  print(reduced)
  # for _ in range(1):
  #   # start = time.time()

  #   # reduced = west_reduce(distribution, max_n)
