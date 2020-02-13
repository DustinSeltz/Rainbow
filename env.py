# -*- coding: utf-8 -*-
from collections import deque
import random
import ai_safety_gridworlds 
import torch


class Env():
  def __init__(self, args): #modified for safety environment
    self.device = args.device
    self.grid = ai_safety_gridworlds.IslandNavigationEnvironment()
    actions = list(ai_safety_gridworlds.environments.shared.Actions())[0:5] # i need to remove one action namely quit
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self): #modified for safety environment
    state = cv2.resize(self.grid.current_game._board, (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self): #modified for safety environment
    # Reset internals
    self._reset_buffer()
    # Reset the World
    self.grid.reset()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action): #modified for safety environment
    # Repeat action 4 times, max pool over last 2 frames
    reward, done = 0, False
    if self.grid.current_game.the_plot.summed_reward is not None:
       reward -= self.grid.current_game.the_plot.summed_reward
    self.grid.step(self.actions.get(action))
    reward += self.grid.current_game.the_plot.summed_reward    
    observation = self._get_state()
    self.state_buffer.append(observation)
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)
  def render(): #modified for safety environment
    pass
  def close(self):
    cv2.destroyAllWindows()
