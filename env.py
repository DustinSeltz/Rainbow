# -*- coding: utf-8 -*-
from collections import deque
import random
import numpy as np
import cv2
from ai_safety_gridworlds.ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import SafetyEnvironment
from ai_safety_gridworlds.ai_safety_gridworlds.environments.shared.safety_game import printTerminations
import torch
import pandas as pd

class Env():
  def __init__(self, args): #modified for safety environment
    self.device = args.device
    self.grid = IslandNavigationEnvironment()
    actions = list(Actions)[0:5] # i need to remove one action namely quit
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self, observation): #modified for safety environment
    # #img = np.zeros((64,64), np.uint8)
    # #font = cv2.FONT_HERSHEY_SIMPLEX
    # #x = 10
    # #y0 = 16
    # #dy = 4
    # #fontScale = 1
    # #lineType = 2
    # #np_ascii_state = self.grid.current_game._board.board
    # #for i, line in enumerate(np_ascii_state) :
       # #y = y0 + i*dy
       # # cv2.putText(img,line, (x,y), font, fontScale, lineType)
    raw_cv_state = cv2.UMat(observation['board'])
    # #84 is a leftover magic number
    state = cv2.resize(raw_cv_state, (84, 84), interpolation=cv2.INTER_LINEAR)
    #127 is the highest ascii code.
    return torch.tensor(state.get(), dtype=torch.float32, device=self.device).div(127)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self): #modified for safety environment
    # Reset internals
    self._reset_buffer()
    # Reset the World & Process and return "initial" state
    _, _, _, raw_observation = self.grid.reset()
    observation = self._get_state(raw_observation)
    self.state_buffer.append(observation)
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action): #modified for safety environment
    #might change code to
    #step_type, reward, discount, observation = self.grid.step(self.actions.get(action))
    step_type, reward , _, raw_observation = self.grid.step(self.actions.get(action))
    if reward is None:
      reward = 0
    print(self.actions.get(action)) #This shows which direction it's moving in. 
    #-self.grid.step(self.actions.get(action))
    #-if self._get_total_reward() is not None:
    #-  reward += self._get_total_reward()
    observation = self._get_state(raw_observation)
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
