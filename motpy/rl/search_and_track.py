import copy
from math import sqrt
import time
from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
from motpy.rfs.tomb import TOMBP
import numpy as np
from motpy.distributions.gaussian import GaussianState
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
    self.extents = np.array([[-1000, 1000],
                             [-1, 1],
                             [-1000, 1000],
                             [-1, 1]], dtype=float)
    self.state_dim = self.extents.shape[0]
    self.obs_dim = self.state_dim + self.state_dim // 2 + 1
    self.volume = np.prod(self.extents[[0, 2], 1] - self.extents[[0, 2], 0])
    self.dt = 1
    self.birth_rate = 1e-3
    self.beamwidth = np.radians(10)
    self.n_expected_init = 10
    self.lambda_c = 1
    self.pg = 0.999
    self.w_min = None
    self.r_min = 1e-4
    self.merge_poisson = True
    self.r_track_loss_threshold = 0.1

    # Define birth distribution
    ngrid = 10
    self.max_set_size = nbirth = ngrid**2
    self.birth_grid = np.meshgrid(np.linspace(*self.extents[0], ngrid),
                                  np.linspace(*self.extents[2], ngrid))
    xgrid, ygrid = self.birth_grid[0].flatten(), self.birth_grid[1].flatten()
    birth_sigmas = np.max(self.extents, axis=1)
    birth_sigmas[[0, 2]] /= ngrid
    self.birth_dist = GaussianState(
        mean=np.stack(
            (xgrid, np.zeros(nbirth), ygrid, np.zeros(nbirth)), axis=1),
        covar=np.broadcast_to(
            np.diag(birth_sigmas), shape=(nbirth, self.state_dim, self.state_dim)),
        weight=np.full(nbirth, self.birth_rate/nbirth),
    )
    self.init_ppp_dist = self.birth_dist

    # Spaces
    self.observation_space = gym.spaces.Dict({
        "tracked": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.max_set_size, self.obs_dim), dtype=np.float32),
        "untracked": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(
            self.max_set_size, self.obs_dim), dtype=np.float32),
        "n_tracked": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=int),
        "n_untracked": gym.spaces.Box(low=0, high=1, shape=(), dtype=int),
    })
    self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))

  def reset(self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    self.np_random = np.random.RandomState(seed)

    # Initialize objects in the environment
    self.ground_truth = []
    n_init = self.np_random.poisson(self.n_expected_init)
    xs = self.np_random.uniform(
        self.extents[:, 0], self.extents[:, 1], size=(n_init, self.state_dim))
    self.ground_truth = [[x] for x in xs]

    # Initialize tracker
    self.tracker = TOMBP(birth_distribution=self.birth_dist,
                         pg=self.pg,
                         w_min=self.w_min,
                         r_min=self.r_min,
                         merge_poisson=self.merge_poisson)
    self.tracker.poisson.distribution = self.init_ppp_dist

    # Initialize models
    cv = ConstantVelocity(ndim_pos=2,
                          q=0.001,
                          position_mapping=[0, 2],
                          velocity_mapping=[1, 3],
                          seed=seed)
    linear = LinearMeasurementModel(
        ndim_state=self.state_dim, covar=np.eye(2), measured_dims=[0, 2], seed=seed)
    self.state_estimator = KalmanFilter(transition_model=cv,
                                        measurement_model=linear)

    obs = self._get_obs(tracker=self.tracker)
    info = {}

    return obs, info

  def step(self, action: np.ndarray):
    # Pd function based on sensor state
    angle = action[0] * (2*np.pi)

    def pd(state):
      # return 0.9
      if isinstance(state, GaussianState):
        x = state.mean
      elif isinstance(state, np.ndarray):
        state = np.atleast_2d(state)
        if state.shape[-1] == 2:
          x = np.zeros((len(state), self.state_dim))
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
      if isinstance(state, GaussianState):
        x = state.mean
      elif isinstance(state, np.ndarray):
        state = np.atleast_2d(state)
        if state.shape[-1] == 2:
          x = np.zeros((len(state), self.state_dim))
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
    Zk = []

    if len(self.ground_truth) > 0:
      # State transition
      states = np.atleast_2d([path[-1] for path in self.ground_truth])
      survived = self.np_random.uniform(size=len(states)) < ps(states)
      states_up = np.empty_like(states)
      states_up[survived] = self.state_estimator.transition_model(
          states[survived], dt=self.dt, noise=True)

      # Object survival
      for i, path in enumerate(self.ground_truth.copy()):
        if survived[i]:
          path.append(states_up[i])
        else:
          self.ground_truth.remove(path)

      detected = self.np_random.uniform(size=len(states_up)) < pd(states_up)
      meas = self.state_estimator.measurement_model(
          states_up[detected], noise=True)
      Zk.extend(list(meas))

    # New object birth
    Nb = self.np_random.poisson(self.birth_rate)
    xb = self.np_random.uniform(
        self.extents[:, 0], self.extents[:, 1], size=(Nb, self.state_dim))
    self.ground_truth.extend([[x] for x in xb])

    # Clutter measurements
    Nc = self.np_random.poisson(self.lambda_c)
    zc = self.np_random.uniform(
        self.extents[[0, 2], 0], self.extents[[0, 2], 1], size=(Nc, 2))
    Zk.extend(list(zc))

    # Update filter state
    self.tracker.mb, self.tracker.poisson = self.tracker.predict(
        state_estimator=self.state_estimator, dt=self.dt, ps_func=ps)
    self.tracker.mb, self.tracker.poisson = self.tracker.update(
        measurements=np.array(Zk), state_estimator=self.state_estimator, lambda_fa=self.lambda_c/self.volume, pd_func=pd)

    obs = self._get_obs(tracker=self.tracker)
    reward = self._get_reward(tracker=self.tracker)
    terminated = False
    truncated = False
    info = {}
    return obs, reward, terminated, truncated, info

  def _get_obs(self, tracker) -> Dict:
    n_mb = len(tracker.mb)
    n_ppp = len(tracker.poisson)

    obs = {}
    obs['n_tracked'] = n_mb
    obs['n_untracked'] = n_ppp
    obs['tracked'] = np.zeros((self.max_set_size, self.obs_dim))
    obs['untracked'] = np.zeros((self.max_set_size, self.obs_dim))

    pos_inds = [0, 2]
    vel_inds = [1, 3]

    if n_mb > 0:
      state_inds = np.arange(n_mb)

      state = tracker.mb.state
      r = tracker.mb.r.reshape(-1, 1)
      means = state.mean
      pos_covar = state.covar[np.ix_(state_inds, pos_inds, pos_inds)]
      covar_diags = np.diagonal(pos_covar, axis1=-2, axis2=-1)
      obs['tracked'][:n_mb] = np.concatenate((means, covar_diags, r), axis=1)

    if n_ppp > 0:
      state_inds = np.arange(n_ppp)
 
      state = tracker.poisson.distribution
      weights = np.log(state.weight.reshape(-1, 1))
      means = state.mean
      pos_covar = state.covar[np.ix_(state_inds, pos_inds, pos_inds)]
      covar_diags = np.diagonal(pos_covar, axis1=-2, axis2=-1)
      obs['untracked'][:n_ppp] = np.concatenate(
          (means, covar_diags, weights), axis=1)

    return obs

  def _get_reward(self, tracker) -> float:
    return 0
    c = 50
    p = 2
    eta = c**p / 2

    # Track loss is the sum of the trace of the covariance matrices of the tracked objects
    pos_inds = [0, 2]
    vel_inds = [1, 3]
    above_thresh = tracker.mb.r > self.r_track_loss_threshold
    if np.any(above_thresh):
      track_inds = np.arange(np.count_nonzero(above_thresh))
      covars = tracker.mb.state[above_thresh].covars
      pos_covars = covars[np.ix_(track_inds, pos_inds, pos_inds)]
      track_loss = np.sum(np.trace(pos_covars, axis1=-2, axis2=-1))
    else:
      track_loss = 0

    # Search loss is the sum
    search_loss = np.sum(tracker.poisson.distribution.weight)

    # Note: For stability reasons, I'm scaling this DOWN by eta
    reward = -(track_loss/eta + search_loss)

    return reward


if __name__ == '__main__':

  env = SearchAndTrackEnv()
  seed = 0
  np.random.seed(seed)
  env.reset(seed=seed)
  env.action_space.seed(seed)

  xmin, xmax = env.extents[0]
  ymin, ymax = env.extents[2]
  ngrid = int(sqrt(len(env.tracker.poisson)))
  xmesh, ymesh = np.meshgrid(
      np.linspace(xmin, xmax, ngrid), np.linspace(ymin, ymax, ngrid))
  grid = np.dstack((xmesh, ymesh))
  fps = 0
  for i in range(1, int(1e3)):
    action = env.action_space.sample()
    # action = np.array([0])
    start = time.time()
    obs, reward, term, trunc, info = env.step(action)
    fps = i / (i + 1) * fps + 1 / (i + 1) * (1 / (time.time() - start))

    plt.clf()
    # intensity = env.tracker.poisson.intensity(
    #     grid=grid, H=env.state_estimator.measurement_model.matrix())

    # plt.imshow(np.log(intensity), extent=(xmin, xmax, ymin, ymax),
    #            origin='lower', aspect='auto')
    # Plot trajectories in env.ground_truth
    # if i > 0:
    #   for path in env.ground_truth:
    #     p = np.array(path)
    #     plt.plot(p[:, 0], p[:, 2], 'r--')
    #   # Plot high-confidence tracks
    #   if np.count_nonzero(env.tracker.mb.r > env.r_estimate_threshold):
    #     for bern in env.tracker.mb[env.tracker.mb.r > env.r_estimate_threshold]:
    #       plt.plot(bern.state.mean[:, 0], bern.state.mean[:, 2], 'k^')
    #   plt.xlim(xmin, xmax)
    #   plt.ylim(ymin, ymax)
    #   # plt.clim([-10, -20])
    #   # plt.colorbar()
    #   plt.draw()
    #   plt.pause(0.001)

    print(f"Action: {action*360}")
    print(f"Reward: {reward}")
    print(f"MB: {obs['tracked'].shape}")
    print(f"PPP: {obs['untracked'].shape}")
    print(
        f"r: {env.tracker.mb.r}")
    print(f"True: {len(env.ground_truth)}")
    print(f"FPS: {fps}")

    print(i)
