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
  if nx == 0:  # All false
    loc_error = 0.0
    miss_error = 0.0
    false_error = c**p/2 * ny
    n_assigned = 0
  elif ny == 0:  # All missed
    loc_error = 0.0
    miss_error = c**p/2 * nx
    false_error = 0.0
    n_assigned = 0
  else:
    # Data association
    x_to_y = jonker.assign2d(C=d, maximize=False)[0]
    x_assigned = np.arange(nx)[x_to_y != -1]
    y_assigned = x_to_y[x_assigned]
    valid = d[x_assigned, y_assigned] < c
    x_assigned, y_assigned = x_assigned[valid], y_assigned[valid]

    # Errors
    n_assigned = len(x_assigned)
    loc_error = np.sum(d[x_assigned, y_assigned]**p)
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
