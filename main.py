# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html

import os
import pc_constants as c

if not os.path.exists(c.SETUP_DONE_FILE):
    exit()

import sys

sys.path.append(os.path.join(os.getcwd(), c.GAME_DIR))
print(sys.path)
os.chdir(c.GAME_DIR)
from game.run import *
from game.pellets import PowerPellet
from collections import deque, namedtuple
from dqn_model import DQN
import torch
import operator
from torch import nn
import math
from random import sample, random, randint
import numpy as np

MEM_SIZE = 100000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000

StateStep = namedtuple("StateStep", ["old_state", "new_state", "action", "reward"])

class PacmanKI:
    def __init__(self):
        self.game = GameController()
        self.mem = deque(maxlen=MEM_SIZE)
        self.dqn = DQN(978, 100, 5, LR, GAMMA)  # .to(DEVICE)
        self.game_count = 0
        self.epsilon = 0
        self.steps_done = 0
        self.last_pallets_eaten = 0

    def train_long(self):
        pass

    def action_step(self, old_state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        if random() >= eps_threshold:
            act = torch.argmax(self.dqn(torch.tensor([old_state], dtype=torch.float))).item()
            print("planned")
        else:
            act = randint(0, 4)
            print("rand")

        self.steps_done += 1

        act = self.set_direction(act)
        new_state, reward, _, _ = self.step(act)
        self.cache(old_state, new_state, act, reward)
        self.dqn.train_step(old_state, new_state, act, reward)
        return new_state

    def step(self, action: int):
        self.game.update()
        return self.eval_state(), self.eval_reward(), False, False

    def eval_state(self):
        # pellet format => (x, y, super, eaten)
        for p in self.game.pallets_eaten:
            p.eaten = True
        pellets = []
        pellets.extend(self.game.pellets.pelletList)
        pellets.extend(self.game.pallets_eaten)
        pellets.sort(key=operator.attrgetter('id'))
        ps = []
        for p in pellets:
            sup = isinstance(p, PowerPellet)
            ps.extend([p.position.x, p.position.y, sup, p.eaten])

        s = [
            self.game.pacman.position.x,
            self.game.pacman.position.y
        ]
        s.extend(ps)
        return s

    def vec2_to_tp(self, vec):
        return [vec.x, vec.y]

    def eval_reward(self):
        r = self.game.pellets.numEaten - self.last_pallets_eaten
        self.last_pallets_eaten = self.game.pellets.numEaten
        print(r)
        return r

    def cache(self, old_state, new_state, action, reward):
        self.mem.append(StateStep(old_state, new_state, action, reward))

    def sample_cache(self):
        if len(self.mem) > BATCH_SIZE:
            return sample(self.mem, BATCH_SIZE)
        else:
            return self.mem


    def set_direction(self, direction: int):
        dirs = [STOP, UP, RIGHT, DOWN, LEFT]

        self.game.pacman.want_direction = dirs[direction]
        return dirs[direction]

    def run(self):
        self.game.startGame()
        self.game.ghosts.ghosts = []
        self.game.pause.setPause(playerPaused=True)
        self.game.update()
        while True:
            self.action_step(self.eval_state())


if __name__ == '__main__':
    pacman_ki = PacmanKI()
    pacman_ki.run()
