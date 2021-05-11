import ctoybox
import os
import gym
import time
from toybox import Toybox
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import DQN


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs('./output/DQN/checkpoint_models/')
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          model.save('./output/DQN/checkpoint_models/dqn_chkpnt_model_{}.pkl'.format(self.num_timesteps))
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                pass
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    pass
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True


timesteps = 10000000
start = time.time()
env_id = 'PacmanToyboxNoFrameskip-v4'
env = gym.make(env_id, alpha=False, grayscale=False)
env = AtariWrapper(env)
log_dir = "./output/DQN/logs/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000000, log_dir=log_dir)

model = DQN("CnnPolicy", env)
model.learn(timesteps, callback = callback)
print('Total Time: {}'.format(time.time() - start))
model.save('./output/DQN/models/dqn_final_model.pkl')
# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "PPO Amidar")
# plt.show()
