# -*- coding: utf-8 -*-
from collections import deque
import random
import numpy as np
import cv2
from ai_safety_gridworlds.ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import Actions
import torch


class Env():
  def __init__(self, args): #modified for safety environment
    self.device = args.device
    self.grid = IslandNavigationEnvironment()
    actions = list(Actions)[0:5] # i need to remove one action namely quit
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self): #modified for safety environment
    np_ascii_state = self.grid.current_game._board.board
    raw_cv_state = cv2.UMat(np.unpackbits(np_ascii_state, axis = 1).astype('uint8'))
    state = cv2.resize(raw_cv_state, (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state.get(), dtype=torch.float32, device=self.device)

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

  def _get_total_reward(self):
    return self.grid.current_game.the_plot._engine_directives.summed_reward
  def step(self, action): #modified for safety environment
    # Repeat action 4 times, max pool over last 2 frames
    reward, done = 0, False
    if self._get_total_reward() is not None:
      reward -= self._get_total_reward()
    else:
      reward = 0
    self.grid.step(self.actions.get(action))
    if self._get_total_reward() is not None:
      reward += self._get_total_reward()
    observation = self._get_state()
    done = not self.grid._game_over
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
  def render(self): #modified for safety environment
    print("Render TODO") #Can reach here using --render --evaluate
  def close(self):
    #cv2.destroyAllWindows()
    self.grid.reset()
