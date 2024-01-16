import copy
from math import sqrt
from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
from motpy.rfs.momb import MOMBP
from motpy.rfs.tomb import TOMBP
import numpy as np
from motpy.distributions.gaussian import GaussianMixture, GaussianState
from motpy.kalman import KalmanFilter
from motpy.models.measurement import LinearMeasurementModel
from motpy.models.transition import ConstantVelocity
import matplotlib.pyplot as plt


def wrap_to_interval(x: Union[float, np.ndarray],
                     start: float,
                     end: float) -> Union[float, np.ndarray]:
  """
  Wrap x to the interval [a, b)

  Args:
      x (Union[float, np.ndarray]): Unwrapped input value
      start (float): Start of interval
      end (float): End of interval

  Returns:
      Union[float, np.ndarray]: _description_
  """
  return (x - start) % (end - start) + start


class SearchAndTrackEnv(gym.Env):
  def __init__(self):
    # Env params
    self.max_set_size = 512
    self.state_dim = 2*4+1
    self.extents = np.array([[-1000, 1000],
                             [-1, 1],
                             [-1000, 1000],
                             [-1, 1]], dtype=float)
    self.volume = np.prod(self.extents[[0, 2], 1] - self.extents[[0, 2], 0])
    self.dt = 1
    self.birth_rate = 1e-2
    # self.ps = 0.999
    self.beamwidth = np.radians(10)
    self.n_expected_init = 10
    self.lambda_c = 1
    self.pg = 0.999
    self.w_min = None
    self.r_min = 1e-4
    self.merge_poisson = True
    self.r_estimate_threshold = 0.5

    # MOMB params
    # Arange birth distribution
    ngrid = 20
    self.birth_grid = np.meshgrid(np.linspace(*self.extents[0], ngrid),
                                  np.linspace(*self.extents[2], ngrid))
    xgrid, ygrid = self.birth_grid[0].flatten(), self.birth_grid[1].flatten()
    # Birth sigmas are uniformly spaced in position, and cover the entire velocity space
    birth_sigmas = np.max(self.extents, axis=1)
    birth_sigmas[[0, 2]] /= ngrid
    self.birth_dist = GaussianMixture(
        means=np.stack(
            (xgrid, np.zeros(ngrid**2), ygrid, np.zeros(ngrid**2)), axis=1),
        covars=np.broadcast_to(
            np.diag(birth_sigmas), shape=(ngrid**2, 4, 4)),
        weights=np.full(ngrid**2, self.birth_rate/ngrid**2),
    )
    self.init_ppp_dist = self.birth_dist

    # Observations are as follows:
    #   - tracked: object state vector, covariance diagonal elements, existence probability
    #   - untracked: object state vector, covariance diagonal elements, Gaussian mixture weight
    self.observation_space = gym.spaces.Dict({
        "tracked": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.max_set_size, self.state_dim), dtype=np.float32),
        "untracked": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.max_set_size, self.state_dim), dtype=np.float32),
        "n_tracked": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=int),
        "n_untracked": gym.spaces.Box(low=0, high=1, shape=(), dtype=int),
    })
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

  def reset(self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    # Initialize objects in the environment
    self.ground_truth = []
    n_init = self.np_random.poisson(self.n_expected_init)
    xs = self.np_random.uniform(self.extents[:, 0], self.extents[:, 1], size=(n_init, 4))
    self.ground_truth = [[x] for x in xs]

    self.tracker = TOMBP(birth_distribution=self.birth_dist,
                         pg=self.pg,
                         w_min=self.w_min,
                         r_min=self.r_min,
                         merge_poisson=self.merge_poisson,
                         r_estimate_threshold=self.r_estimate_threshold)
    self.tracker.poisson.distribution = self.init_ppp_dist

    cv = ConstantVelocity(ndim_pos=2,
                          q=0.001,
                          position_mapping=[0, 2],
                          velocity_mapping=[1, 3],
                          seed=seed)
    linear = LinearMeasurementModel(
        ndim_state=4, covar=np.eye(2), measured_dims=[0, 2], seed=seed)
    self.state_estimator = KalmanFilter(transition_model=cv,
                                        measurement_model=linear)

    obs = self._get_obs(tracker=self.tracker)
    info = {}

    return obs, info

  def step(self, action: np.ndarray):
    # Pd function based on sensor state
    angle = action[0] * (2*np.pi)

    def pd(state):
      if isinstance(state, GaussianMixture):
        x = state.means
      elif isinstance(state, np.ndarray):
        state = np.atleast_2d(state)
        if state.shape[-1] == 2:
          x = np.zeros((len(state), 4))
          x[:, [0, 2]] = state
        else:
          x = state

      obj_angle = np.arctan2(x[:, 2], x[:, 0])
      # Check if object is within beamwidth
      angle_diff = wrap_to_interval(angle-obj_angle, -np.pi, np.pi)

      out = np.zeros(len(x))
      out[np.abs(angle_diff) < self.beamwidth/2] = 0.8
      out[(np.abs(angle_diff) > self.beamwidth/2) &
          (np.abs(angle_diff) < self.beamwidth)] = 0.4
      return out

    def ps(state):
      if isinstance(state, GaussianMixture):
        x = state.means
      elif isinstance(state, np.ndarray):
        state = np.atleast_2d(state)
        if state.shape[-1] == 2:
          x = np.zeros((len(state), 4))
          x[:, [0, 2]] = state
        else:
          x = state

      out = np.zeros(len(x))
      xmin, ymin = self.extents[[0, 2], 0]
      xmax, ymax = self.extents[[0, 2], 1]
      out[np.logical_and(x[:, 0] >= xmin, x[:, 0] <= xmax) &
          np.logical_and(x[:, 1] >= ymin, x[:, 2] <= ymax)] = 1.0
      return out

    # Collect measurements from existing objects
    survived = np.ones(len(self.ground_truth), dtype=bool)
    Zk = []
    for i, path in enumerate(self.ground_truth):

      if self.np_random.uniform() > ps(path[-1]):
        survived[i] = False
        continue
      path.append(self.state_estimator.transition_model(
          path[-1], dt=self.dt, noise=False))

      if self.np_random.uniform() < pd(path[-1]):
        Zk.append(self.state_estimator.measurement_model(
            path[-1], noise=True))

    self.ground_truth = [path for i, path in enumerate(
        self.ground_truth) if survived[i]]

    # Clutter measurements
    Nc = self.np_random.poisson(self.lambda_c)
    zc = self.np_random.uniform(
        self.extents[[0, 2], 0], self.extents[[0, 2], 1], size=(Nc, 2))
    Zk.extend(list(zc))

    # New object birth
    Nb = self.np_random.poisson(self.birth_rate)
    xb = self.np_random.uniform(
        self.extents[:, 0], self.extents[:, 1], size=(Nb, 4))
    for x in xb:
      self.ground_truth.append([x])

    Zk = np.array(Zk)

    # Update filter state
    self.tracker.mb, self.tracker.poisson = self.tracker.predict(
        state_estimator=self.state_estimator, dt=self.dt, ps_func=ps)
    self.tracker.mb, self.tracker.poisson = self.tracker.update(
        measurements=Zk, state_estimator=self.state_estimator, lambda_fa=self.lambda_c/self.volume, pd_func=pd)

    obs = self._get_obs(tracker=self.tracker)
    reward = self._get_reward(momb=self.tracker)
    terminated = False
    truncated = False
    info = {}
    return obs, reward, terminated, truncated, info

  def _get_obs(self, tracker) -> Dict:
    return None  # TODO: Fix obs length
    obs = {}
    obs['n_tracked'] = len(tracker.mb)
    obs['n_untracked'] = len(tracker.poisson)

    obs['tracked'] = np.empty(
        self.observation_space['tracked'].shape, dtype=np.float32)
    obs['untracked'] = np.empty(
        self.observation_space['untracked'].shape, dtype=np.float32)

    for i, bern in enumerate(tracker.mb):
      obs['tracked'][i] = np.append(self._state_to_obs(bern.state), bern.r)

    for i, (w, state) in enumerate(tracker.poisson):
      obs['untracked'][i] = np.append(self._state_to_obs(state), w)

    return obs

  def _get_reward(self, momb) -> float:
    return 0
    c = 50
    p = 2
    eta = c**p / 2
    # Track loss is the sum of the trace of the covariance matrices of the tracked objects

    track_loss = np.sum([np.trace(bern.state.covar)
                        for bern in momb.mb if bern.r > self.r_estimate_threshold])

    # TODO: Bostrom rost reward
    return 0

  def _state_to_obs(self, state: GaussianState) -> np.ndarray:
    mean = state.mean
    covar_diag = np.diag(state.covar)
    return np.concatenate([mean, covar_diag])


if __name__ == '__main__':

  env = SearchAndTrackEnv()
  seed = 0
  np.random.seed(seed)
  env.reset(seed=seed)

  xmin, xmax = env.extents[0]
  ymin, ymax = env.extents[2]
  ngrid = int(sqrt(len(env.tracker.poisson)))
  xmesh, ymesh = np.meshgrid(
      np.linspace(xmin, xmax, ngrid), np.linspace(ymin, ymax, ngrid))
  grid = np.dstack((xmesh, ymesh))

  for i in range(int(1e5)):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)

    

    plt.clf()
    intensity = env.tracker.poisson.intensity(
        grid=grid, H=env.state_estimator.measurement_model.matrix())

    plt.imshow(intensity, extent=(xmin, xmax, ymin, ymax),
               origin='lower', aspect='auto')
    # Plot trajectories in env.ground_truth
    for path in env.ground_truth:
      p = np.array(path)
      plt.plot(p[:, 0], p[:, 2], 'r--')
    # Plot high-confidence tracks
    if np.count_nonzero(env.tracker.mb.r > env.r_estimate_threshold):
      for bern in env.tracker.mb[env.tracker.mb.r > env.r_estimate_threshold]:
        plt.plot(bern.state.means[:, 0], bern.state.means[:, 2], 'k^')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # # plt.clim([0, 1e-6])
    # # plt.colorbar()
    plt.draw()
    plt.pause(0.001)
    
    print(f"Action: {action*360}")
    print(f"PPP: {len(env.tracker.poisson)}")
    print(
        f"MB: {len(env.tracker.mb)}")
    print(
        f"r: {env.tracker.mb.r}")
    print(f"True: {len(env.ground_truth)}")
    # print(f"Undetected: {np.sum(env.momb.poisson.weights)}")

    print(i)

    # plt.draw()
