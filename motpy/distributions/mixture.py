from typing import Optional, Tuple
import numpy as np
from motpy.distributions.gaussian import Gaussian, merge_gaussians


from motpy.distributions.gaussian import Gaussian
import numpy as np
from motpy.distributions.gaussian import merge_gaussians


def runnals_cost(wi: np.ndarray,
                 wj: np.ndarray,
                 Pi: np.ndarray,
                 Pj: np.ndarray,
                 Pij: np.ndarray,
                 ) -> np.ndarray:
  return 0.5 * (
      (wi + wj) * np.log(np.linalg.det(Pij))
      - np.expand_dims(wi * np.log(np.linalg.det(Pi)), -1)
      - np.expand_dims(wj * np.log(np.linalg.det(Pj)), -2)
  )


def runnalls_merge(state: Gaussian, num_desired: int) -> Gaussian:
  num_components = state.size

  w, mu, P = state.weight.copy(), state.mean.copy(), state.covar.copy()

  # Step 1: Compute all possible mixture pairs and their costs
  ij = np.indices((num_components, num_components)).transpose(1, 2, 0)
  w_mix, mu_mix, P_mix = merge_gaussians(mu[ij], P[ij], w[ij])
  c = runnals_cost(wi=w, wj=w, Pi=P, Pj=P, Pij=P_mix)
  c[np.diag_indices_from(c)] = np.inf

  # Step 2: While we still need to merge...
  valid = np.ones(num_components, dtype=bool)
  for i in range(num_components - num_desired):
    # Step 3: Select the merge hypothesis with the smallest cost
    i, j = np.unravel_index(np.argmin(c), c.shape)
    w[i], mu[i], P[i] = w_mix[i, j], mu_mix[i, j], P_mix[i, j]

    # Step 4: Update mixture hypotheses and costs for new component
    w_mix[i], mu_mix[i], P_mix[i] = w_mix[:, i], mu_mix[:, i], P_mix[:, i] = merge_gaussians(
        mu[ij[i]], P[ij[i]], w[ij[i]]
    )
    c[i] = c[:, i] = np.where(
        ~np.isinf(c[i]),
        runnals_cost(wi=w[i], wj=w, Pi=P[i], Pj=P, Pij=P_mix[i]),
        np.inf,
    )

    # Step 5: Remove merged component
    c[j] = c[:, j] = np.inf
    valid[j] = False
    
  return Gaussian(mean=mu[valid], covar=P[valid], weight=w[valid])


def static_reduce(distribution: Gaussian) -> Gaussian:

  n = distribution.size
  state = distribution[:n//2]
  birth_state = distribution[n//2:]
  wmix = np.stack((state.weight, birth_state.weight), axis=0)
  wmix /= np.sum(wmix + 1e-15, axis=0)
  xmix = np.stack((state.mean, birth_state.mean), axis=0)
  Pmix = np.stack((state.covar, birth_state.covar), axis=0)
  merged_state = Gaussian(
      mean=np.einsum('i..., i...j -> ...j', wmix, xmix),
      covar=np.einsum('i..., i...jk -> ...jk', wmix, Pmix),
      weight=state.weight + birth_state.weight,
  )

  return merged_state


if __name__ == '__main__':
  state = Gaussian(
      mean=np.array([np.zeros(4), np.ones(4), np.ones(4)]),
      covar=np.array([np.eye(4), 2*np.eye(4), 3*np.eye(4)]),
      weight=np.ones(3)
  )
  new_state = runnalls_merge(state, 1)
  print(new_state)
