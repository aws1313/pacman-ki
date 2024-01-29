# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

import os
import pc_constants as c

if not os.path.exists(c.SETUP_DONE_FILE):
    exit()

import sys

sys.path.append(os.path.join(os.getcwd(), c.GAME_DIR))
print(sys.path)

from game.run import *
from collections import deque
from dqn_model import DQN
import torch
from torch import nn
import numpy as np

MEM_SIZE = 100

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


class PacmanKI:
    def __init__(self):
        self.game = GameController()
        self.mem = deque(maxlen=MEM_SIZE)
        self.dqn = DQN(2, 2, 5)# .to(DEVICE)

    def train_long(self):
        pass

    def train_step(self, old_state):
        act = self.dqn(torch.tensor(np.array([old_state, old_state], dtype=np.int64), dtype=torch.float))
        print(act)
        act = act.argmax().item()
        print(act)
        new_state,_,_,_ = self.step(act)
        return new_state



    def step(self, action: int):
        self.set_direction(action)
        self.game.update()
        return self.eval_state(), self.eval_reward(), False, False

    def eval_state(self):
        print(self.game.pacman.position)
        return self.vec2_to_tp(self.game.pacman.position)

    def vec2_to_tp(self, vec):
        return [vec.x, vec.y]

    def eval_reward(self):
        return self.game.pellets.numEaten

    def cache(self):
        pass

    def set_direction(self, direction: int):
        dirs = [STOP, UP, RIGHT, DOWN, LEFT]

        self.game.pacman.want_direction = dirs[direction]

    def run(self):
        self.game.startGame()
        self.game.update()
        while True:
            self.train_step(self.eval_state())




if __name__ == '__main__':
    print(f"Running on: {DEVICE}")

    os.chdir(c.GAME_DIR)
    pacman_ki = PacmanKI()
    pacman_ki.run()
