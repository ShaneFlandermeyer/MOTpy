import copy
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
    self.extents = np.array([[-100, 100],
                             [-1, 1],
                             [-100, 100],
                             [-1, 1]])
    self.volume = np.prod(self.extents[[0, 2], 1] - self.extents[[0, 2], 0])
    self.dt = 1
    self.birth_rate = 1e-2
    self.ps = 0.999
    self.beamwidth = 2*np.pi/10
    self.n_expected_init = 10
    self.lambda_c = 0

    # MOMB params
    # Arange birth distribution
    ngrid = 30
    birth_grid = np.meshgrid(np.linspace(-100, 100, ngrid),
                             np.linspace(-100, 100, ngrid))
    xgrid, ygrid = birth_grid[0].flatten(), birth_grid[1].flatten()
    self.birth_dist = GaussianMixture(
        means=np.array([[x, 0, y, 0] for x, y in zip(xgrid, ygrid)]),
        covars=(np.diag([100/ngrid/1, 1, 100/ngrid/1, 1])[None, ...]
               ** 2).repeat(ngrid**2, axis=0),
        weights=np.full(ngrid**2, self.birth_rate/ngrid**2),
    )

    self.pg = 0.999
    self.w_min = None
    self.r_min = 1e-4
    self.merge_poisson = True
    self.r_estimate_threshold = 0.5
    self.init_ppp_dist = copy.deepcopy(self.birth_dist)

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
    for i in range(n_init):
      x = self.np_random.uniform(self.extents[:, 0], self.extents[:, 1])
      self.ground_truth.append([x])

    self.momb = TOMBP(birth_distribution=self.birth_dist,
                      pg=self.pg,
                      w_min=self.w_min,
                      r_min=self.r_min,
                      merge_poisson=self.merge_poisson,
                      r_estimate_threshold=self.r_estimate_threshold)
    self.momb.poisson.distribution = self.init_ppp_dist
    cv = ConstantVelocity(ndim_pos=2,
                          q=0.001,
                          position_mapping=[0, 2],
                          velocity_mapping=[1, 3],
                          seed=seed)
    linear = LinearMeasurementModel(
        ndim_state=4, covar=np.eye(2), measured_dims=[0, 2], seed=seed)
    self.state_estimator = KalmanFilter(transition_model=cv,
                                        measurement_model=linear)

    obs = self._get_obs(momb=self.momb)
    info = {}

    return obs, info

  def step(self, action: np.ndarray):
    # Pd function based on sensor state
    angle = action[0] * (2*np.pi)

    def pd(state):
      # return 0.9
      x = np.atleast_2d(state) if isinstance(state, np.ndarray) else state.means
      obj_angle = np.arctan2(x[:, 2], x[:, 0])
      # Check if object is within beamwidth
      angle_diff = wrap_to_interval(angle-obj_angle, -np.pi, np.pi)

      out = np.zeros(len(x))
      out[np.abs(angle_diff) < self.beamwidth/2] = 1.0
      out[np.logical_and(np.abs(angle_diff) > self.beamwidth/2,
          np.abs(angle_diff) < self.beamwidth)] = 0.5
      return out

    # Collect measurements from existing objects
    alive = np.ones(len(self.ground_truth), dtype=bool)
    Zk = []
    for i, path in enumerate(self.ground_truth):
      if self.np_random.uniform() > self.ps:
        alive[i] = False
        continue
      if not np.all(np.logical_and(self.extents[:, 0] <= path[-1],
                                   path[-1] <= self.extents[:, 1])):
        path[-1][[1, 3]] *= -1

      path.append(self.state_estimator.transition_model(
          path[-1], dt=self.dt, noise=False))
      if self.np_random.uniform() < pd(path[-1]):
        Zk.append(self.state_estimator.measurement_model(
            path[-1], noise=True))
    self.ground_truth = [path for i, path in enumerate(
        self.ground_truth) if alive[i]]

    # Clutter measurements
    # TODO: Forcing Nc > 0
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
    self.momb.mb, self.momb.poisson = self.momb.predict(
        state_estimator=self.state_estimator, dt=self.dt, ps=self.ps)
    self.momb.mb, self.momb.poisson = self.momb.update(
        measurements=Zk, state_estimator=self.state_estimator, lambda_fa=self.lambda_c/self.volume, pd=pd)

    obs = self._get_obs(momb=self.momb)
    reward = self._get_reward(momb=self.momb)
    terminated = False
    truncated = False
    info = {}
    return obs, reward, terminated, truncated, info

  def _get_obs(self, momb) -> Dict:
    return None  # TODO: Fix obs length
    obs = {}
    obs['n_tracked'] = len(momb.mb)
    obs['n_untracked'] = len(momb.poisson)

    obs['tracked'] = np.empty(
        self.observation_space['tracked'].shape, dtype=np.float32)
    obs['untracked'] = np.empty(
        self.observation_space['untracked'].shape, dtype=np.float32)

    for i, bern in enumerate(momb.mb):
      obs['tracked'][i] = np.append(self._state_to_obs(bern.state), bern.r)

    for i, (w, state) in enumerate(momb.poisson):
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
  seed = 0
  np.random.seed(seed)

  env = SearchAndTrackEnv()
  env.reset()
  xmin, xmax = env.extents[0]
  ymin, ymax = env.extents[2]
  xmesh, ymesh = np.meshgrid(
      np.linspace(xmin, xmax, 25), np.linspace(ymin, ymax, 25))
  grid = np.dstack((xmesh, ymesh))

  # plt.figure()
  # action = np.array([0.0])
  for i in range(int(1e5)):
    # action = (action + 0.05) % 1
    action = env.action_space.sample()
    # action = np.array([0.25])
    # Steer to a random target
    # if i % 1 == 0:
    # object_ind = np.random.randint(len(env.ground_truth))
    # object_path = env.ground_truth[object_ind]
    # object_pos = object_path[-1][[0, 2]]
    # angle = np.arctan2(object_pos[1], object_pos[0])
    # action = np.array([angle/(2*np.pi)])

    obs, reward, term, trunc, info = env.step(action)

    print(f"Action: {action*360}")

    # plt.clf()
    # # intensity = env.momb.poisson.intensity(
    # #     grid=grid, H=env.state_estimator.measurement_model.matrix())

    # # plt.imshow(intensity, extent=(xmin, xmax, ymin, ymax),
    # #            origin='lower', aspect='auto')
    # # Plot trajectories in env.ground_truth
    # for path in env.ground_truth:
    #   p = np.array(path)
    #   plt.plot(p[:, 0], p[:, 2], 'r--')
    # # Plot high-confidence tracks
    # if np.count_nonzero(env.momb.mb.r > env.r_estimate_threshold):
    #   for bern in env.momb.mb[env.momb.mb.r > env.r_estimate_threshold]:
    #     plt.plot(bern.state.means[:, 0], bern.state.means[:, 2], 'k^')

    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)

    # # plt.clim([0, 1e-6])
    # # plt.colorbar()
    # plt.draw()
    # plt.pause(0.001)
    print(f"PPP: {len(env.momb.poisson)}")
    print(
        f"MB: {len([bern for bern in env.momb.mb if bern.r > env.r_estimate_threshold])}")
    print(f"r: {env.momb.mb.r[env.momb.mb.r > env.r_estimate_threshold]}")
    print(f"True: {len(env.ground_truth)}")
    # print(f"Undetected: {np.sum(env.momb.poisson.weights)}")

    print(i)

    # plt.draw()
