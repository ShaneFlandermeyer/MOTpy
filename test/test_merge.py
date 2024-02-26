import numpy as np
import jax
import jax.numpy as jnp
from motpy.measures import pairwise_euclidean

# @jax.jit


def cluster(x, w, threshold):
  # Compute pairwise distances between all points
  dists = pairwise_euclidean(x, x)
  mask = dists < threshold

  # Sort distance matrix rows by weight
  inds = jnp.argsort(w, descending=True)
  dists = dists[inds]
  mask = mask[inds]

  # Determine cluster indices
  cluster_inds = jnp.full(x.shape[0], -1, dtype=jnp.int32)
  cluster_count = 0
  for i in range(x.shape[0]):
    # Greedily assign all nearby points to this cluster. We only increment the cluster count if this cluster includes unused points.
    cluster_inds = jnp.where(mask[i], cluster_count, cluster_inds)
    cluster_count += jnp.any(mask[i])

    # Mark points in this cluster as used
    mask = jnp.where(mask[i], False, mask)

  return cluster_inds


def merge_mixture(means: np.ndarray,
                  covars: np.ndarray,
                  weights: np.ndarray,
                  threshold: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Compute pairwise distances between all points
  dists = pairwise_euclidean(means, means)
  mask = dists < threshold

  # Sort distance matrix rows by weight
  inds = jnp.argsort(weights, descending=True)
  dists = dists[inds]
  mask = mask[inds]

  # Determine cluster indices and normalized mixture weights
  mix_weights = jnp.empty((means.shape[0], weights.size))
  cluster_inds = jnp.empty(means.shape[0], dtype=jnp.int32)
  cluster_count = 0
  for i in range(means.shape[0]):
    # Compute (unnormalized) mixture weights for this cluster
    mix_weights = mix_weights.at[i].set(jnp.where(mask[i], weights, 0))

    # Greedily assign all nearby points to this cluster. We only increment the cluster count if this cluster includes unused points.
    cluster_inds = jnp.where(mask[i], cluster_count, cluster_inds)
    cluster_count += jnp.any(mask[i])

    # Mark points in this cluster as used
    mask = jnp.where(mask[i], False, mask)

  # Match moments for all clusters
  x = means
  P = covars
  w = mix_weights / (jnp.sum(mix_weights, axis=-1, keepdims=True) + 1e-15)
  mix_means = jnp.einsum('...i, ...ij -> ...j', w, x)
  mix_covars = jnp.einsum('...i, ...ijk->...jk', w, P)
  mix_covars += jnp.einsum('...i,...ij,...ik->...jk', w, x, x)
  mix_covars -= jnp.einsum('...i,...j->...ij', mix_means, mix_means)

  return mix_means, mix_covars, mix_weights, cluster_inds



if __name__ == '__main__':
  threshold = 1e-15

  # Create data
  x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
  x = x.ravel()
  y = y.ravel()
  P = np.eye(2)[None, ...].repeat(128, 0)
  means = np.stack((x, y), axis=-1)
  means = np.concatenate((means, means))
  weights = np.concatenate((np.ones(64), 0.01*np.ones(64)))

  #
  # cluster_inds = cluster(means, weights, threshold)
  merge(means, P, weights, threshold)
  # print(cluster_inds)
