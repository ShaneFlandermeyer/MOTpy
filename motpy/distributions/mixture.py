import numpy as np
import numba as nb
from motpy.distributions.gaussian import Gaussian

def runnalls(dist, num_components):
  valid = np.ones(dist.size, dtype=bool)
  inds = np.arange(dist.size)
  w = dist.weight
  mu = dist.mean
  P = dist.covar


  while np.count_nonzero(valid) > num_components:
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
    c = (0.5 * wij * np.log(np.linalg.det(Pij))) - \
        (wi * np.log(np.linalg.det(Pi))) - \
        (0.5 * wj * np.log(np.linalg.det(Pj)))

    # Merge the two components with the smallest cost
    # Set the diagonal to infinity so that we don't merge a component with itself
    c[np.diag_indices_from(c)] = np.inf
    i, j = np.unravel_index(np.argmin(c), c.shape)
    valid_inds = inds[valid]
    w[valid_inds[i]] = wij[i, j]
    mu[valid_inds[i]] = mu_ij[i, j]
    P[valid_inds[i]] = Pij[i, j]
    valid[valid_inds[j]] = False

  dist = Gaussian(
    mean=mu[valid],
    covar=P[valid],
    weight=w[valid]
  )
  return dist

if __name__ == '__main__':
  n = 128
  dist = Gaussian(
      mean=np.ones((n, 4)),
      covar=np.eye(4)[None, ...].repeat(n, axis=0),
      weight=np.ones(n) / n
  )
  runnalls(dist, 9)
  
  import time
  start = time.time()
  for i in range(100):
    runnalls(dist, 64)
  print((time.time() - start) / 100)
