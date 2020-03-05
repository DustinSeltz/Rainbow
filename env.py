# -*- coding: utf-8 -*-
from collections import deque
import random
import numpy as np
import cv2
from ai_safety_gridworlds.ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.ai_safety_gridworlds.environments.island_navigation_intervention import IslandNavigationEnvironmentIntervention
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import SafetyEnvironment
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import printTerminations
import torch
import pandas as pd

class Env():
  def __init__(self, args): #modified for safety environment
    self.device = args.device
    if args.intervention:
      self.grid = IslandNavigationEnvironmentIntervention()
    else:
      self.grid = IslandNavigationEnvironment()
    actions = list(Actions)[0:5] # i need to remove one action namely quit
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self): #modified for safety environment
    np_ascii_state = self.grid.current_game._board.board
    raw_cv_state = cv2.UMat(np_ascii_state)
    #84 is a leftover magic number
    state = cv2.resize(raw_cv_state, (84, 84), interpolation=cv2.INTER_LINEAR)
    #127 is the highest ascii code.
    return torch.tensor(state.get(), dtype=torch.float32, device=self.device).div(127)

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
    #might change code to
    #step_type, reward, discount, observation = self.grid.step(self.actions.get(action))
    step_type, reward , _, _ = self.grid.step(self.actions.get(action))
    if reward is None:
      reward = 0
    #print(self.actions.get(action)) #This shows which direction it's moving in. 
    #-self.grid.step(self.actions.get(action))
    #-if self._get_total_reward() is not None:
    #-  reward += self._get_total_reward()
    observation = self._get_state()
    done = self.grid.current_game.the_plot._engine_directives.game_over or step_type.last()
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
    #print("Render TODO") #Can reach here using --render --evaluate
    #See safety_game.py in Rainbow\ai_safety_gridworlds\ai_safety_gridworlds\environments\shared 
    #print("Overall performance:", self.grid.get_overall_performance()) #Okay, this prints None, the default value. So we're never modifying _calculate_overall_performance?
    #print("Some hidden reward:", self.grid._get_hidden_reward()) #And this is all -1.
    #TODO render ASCII art of board
    #np.array([chr(x) for x in range(127)])[a]
    ascii_index = self.grid.current_game._board.board
    ascii_val = np.array([chr(x) for x in range(127)])[ascii_index]
    ascii_nice = pd.DataFrame(ascii_val)
    print(ascii_val, flush = True)
    printTerminations()
  def close(self):
    #cv2.destroyAllWindows()
    self.grid.reset()
