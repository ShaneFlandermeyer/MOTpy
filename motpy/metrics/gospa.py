import numpy as np
from motpy.association import jonker


def gospa(
    X: np.ndarray,
    Y: np.ndarray,
    d: np.ndarray,
    c: float,
    p: int = 1,
) -> float:
  """
  GOSPA with alpha = 2

  Parameters
  ----------
  X : np.ndarray
      Ground truth states
  Y : np.ndarray
      State estimates
  d : np.ndarray
      Pairwise distance between X and Y
  c : float
      Cut-off distance
  p : int, optional
      GOSPA exponent, by default 1

  Returns
  -------
  float
      _description_
  """
  nx, ny = len(X), len(Y)
  if nx == 0:  # No ground truth
    loc_error = 0.0
    miss_error = 0.0
    false_error = c**p/2 * ny
  elif ny == 0:  # All missed
    loc_error = 0.0
    miss_error = c**p/2 * nx
    false_error = 0.0
  else:
    # Data association
    x_to_y = jonker.assign2d(C=d, maximize=False)[0]

    # Localization error
    assigned_x = np.arange(nx)[x_to_y != -1]
    assigned_y = x_to_y[assigned_x]
    valid_x = assigned_x[d[assigned_x, assigned_y] < c]
    valid_y = assigned_y[d[assigned_x, assigned_y] < c]
    n_assigned = len(valid_x)
    loc_error = np.sum(d[valid_x, valid_y]**p)

    # Cardinality error
    miss_error = c**p/2 * (nx - n_assigned)
    false_error = c**p/2 * (ny - n_assigned)

  return dict(
      GOSPA=(loc_error + miss_error + false_error)**(1/p),
      loc_error=loc_error,
      miss_error=miss_error,
      false_error=false_error,
      n_assigned=n_assigned,
  )


if __name__ == '__main__':
  # Ground truth
  X = np.array([
      [1, 1],
      [1, 5],
      [10, 10],
  ])
  # Estimates
  Y = np.array([
      [1, 0.9],
      [7, 2],
      [7, 0.5],
  ])

  d = np.linalg.norm(X[:, None] - Y[None, :], ord=2, axis=-1)

  print(gospa(X=X, Y=Y, d=d, c=1.0, p=1))
