
from time import time
import numpy as np
from typing import Tuple


def assign2d(
    C: np.ndarray,
    maximize: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
  """
  Solve the two-dimensional assignment problem with a rectangular cost matrix C.
  Uses a modified Jonker-Volgenant algorithm.

  Parameters
  ----------
  C : ndarray
      num_row x num_col cost matrix. Forbidden assignments are indicated by +Inf for minimization and -Inf for maximization.
  maximize : bool, optional
      If True, assignments maximize the cost in C. By default False. 

  Returns
  -------
  row_assignments : ndarray
      Row assignment vector which indicates the column assigned to each row. -1 means the row is unassigned. Empty if the assignment is infeasible.
  col_assignments : ndarray
      Column assignment vector which indicates the row assigned to each column. -1 means the column is unassigned. Empty if the assignment is infeasible.
  gain : float
      Sum of assigned elements in C. -1 if infeasible.
  u : ndarray
      Dual variable for columns.
  v : ndarray
      Dual variable for rows.
  """
  C = np.asarray(C)
  num_row, num_col = C.shape

  transposed = False
  if num_row > num_col:
    C = C.T
    transposed = True
    num_row, num_col = num_col, num_row

  # Make all elements non-negative for assignment algorithm
  if maximize:
    C = -C
  C_delta = np.min(C).clip(None, 0)
  C = C - C_delta

  row_assignments = np.full(num_row, -1, dtype=int)
  col_assignments = np.full(num_col, -1, dtype=int)
  u = np.zeros(num_row)
  v = np.zeros(num_col)

  for irow in range(num_row):
    sink, path, u, v = shortest_path(
        irow, u, v, C, row_assignments, col_assignments
    )
    if sink == -1:  # Infeasible
      return None, None, -1, u, v

    # Augment the previous solution
    j = sink
    while True:
      i = path[j]
      col_assignments[j] = i
      j, row_assignments[i] = row_assignments[i], j
      if i == irow:  # Continue until we reach the beginning of the path
        break

  # Calculate gain
  gain = 0
  for irow in range(num_col):
    gain += C[col_assignments[irow], irow]

  # Adjust gain for initial offset
  if maximize:
    gain = -gain + C_delta * num_col
    u, v = -u, -v
  else:
    gain = gain + C_delta * num_col

  if transposed:
    col_assignments, row_assignments = row_assignments, col_assignments
    u, v = v, u

  return row_assignments, col_assignments, gain, u, v


def shortest_path(
    row_index: int,
    u: np.ndarray,
    v: np.ndarray,
    C: np.ndarray,
    row_assignments: np.ndarray,
    col_assignments: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
  num_row, num_col = C.shape
  path = np.zeros(num_col, dtype=int)
  shortest_path_costs = np.full(num_col, np.inf)
  scanned_rows = np.zeros(num_row, dtype=bool)
  scanned_cols = np.zeros(num_col, dtype=bool)

  sink = -1
  min_val = 0
  i = row_index
  while sink == -1:
    # Update scanned row set
    scanned_rows[i] = True

    # Update path and shortest path costs
    reduced_cost = min_val + C[i, ~scanned_cols] - u[i] - v[~scanned_cols]
    path[~scanned_cols] = np.where(
        reduced_cost < shortest_path_costs[~scanned_cols],
        i, path[~scanned_cols]
    )
    shortest_path_costs[~scanned_cols] = np.minimum(
        shortest_path_costs[~scanned_cols], reduced_cost
    )

    # Select shortest path column or determine infeasibility
    unscanned_col_inds = np.where(~scanned_cols)[0]
    shortest_unscanned_col_ind = np.argmin(shortest_path_costs[~scanned_cols])
    j = unscanned_col_inds[shortest_unscanned_col_ind]
    if shortest_path_costs[j] == np.inf:
      return None, None, u, v  # Infeasible assignment

    # Update scanned column set
    scanned_cols[j] = True
    min_val = shortest_path_costs[j]

    # Continue until we hit an unassigned column
    if col_assignments[j] == -1:
      sink = j
    else:
      i = col_assignments[j]

  # Update dual variables
  u[row_index] += min_val
  mask = scanned_rows
  mask[row_index] = False
  u[mask] += min_val - shortest_path_costs[row_assignments[mask]]
  v[scanned_cols] += -min_val + shortest_path_costs[scanned_cols]

  return sink, path, u, v


if __name__ == '__main__':
  import time

  start = time.time()
  N = 10
  for i in range(N):
    print(i)
    C = np.random.uniform(0, 1, (200, 200))
    assign2D(C, maximize=False)
  print('Time:', 1/N*(time.time() - start))
