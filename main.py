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
import math
from random import sample, random, randint
import numpy as np
import datetime

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
        self.dqn = DQN(2016, 16, 4, LR, GAMMA)  # .to(DEVICE)
        self.game_count = 0
        self.epsilon = 0
        self.steps_done = 0
        self.last_pallets_eaten = 0
        self.last_pac_pos = None

    def train_long(self):
        cache = self.sample_cache()
        old_states, new_states, actions, rewards = zip(*cache)

        self.dqn.train_step(np.array(old_states), np.array(new_states), actions, rewards)

    def action_step(self, old_state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        if random() >= eps_threshold:
            action = torch.argmax(self.dqn(torch.from_numpy(old_state))).item()
            #print("planned")
        else:
            action = randint(0, 3)
            #print("rand")

        self.steps_done += 1

        action = self.set_direction(action)
        new_state, reward, died, won = self.step()
        self.cache(old_state, new_state, action, reward)
        self.dqn.train_step(old_state, new_state, action, reward)
        return died, won

    def step(self, ):
        self.game.update()
        died = False
        if self.game.pacman_died:
            self.game.pacman_died = False
            died = True
        # new_state, reward, died, won
        return self.eval_state(), self.eval_reward(died, False), died, False

    def tile_from_pos(self, x, y):
        x = round((x + 4) / 16)
        y = round(y / 16)
        return x, y

    def eval_state(self):
        s = np.full((36, 56), 0, dtype=np.float32)

        for i in self.game.pellets.pelletList:
            tx, ty = self.tile_from_pos(i.position.x, i.position.y)
            s[ty][ty] = 0.5

        for i in self.game.pallets_eaten:
            tx, ty = self.tile_from_pos(i.position.x, i.position.y)
            s[ty][ty] = 0.4

        tx, ty = self.tile_from_pos(self.game.pacman.position.x, self.game.pacman.position.y)
        s[ty][tx] = 1

        for g in self.game.ghosts.ghosts:
            tx, ty = self.tile_from_pos(g.position.x, g.position.y)
            s[ty][tx] = 0.1

        s[0][0] = self.game.pacman.direction + 2
        return s.flatten()
        # pellet format => (x, y, super, eaten)
        for p in self.game.pallets_eaten:
            p.eaten = True
        pellets = []
        # print(len(self.game.pallets_eaten))
        # print(len(self.game.pellets.pelletList))
        pellets.extend(self.game.pallets_eaten)
        pellets.extend(self.game.pellets.pelletList)
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
        print(s)
        return torch.tensor(s, dtype=torch.float)

    def vec2_to_tp(self, vec):
        return [vec.x, vec.y]

    def eval_reward(self, died, won):
        pl = (self.game.pellets.numEaten - self.last_pallets_eaten) * 10
        self.last_pallets_eaten = self.game.pellets.numEaten
        s = 0
        # s = -1 if self.last_pac_pos==self.game.pacman.position else 0

        self.last_pac_pos = self.game.pacman.position

        r = pl + s

        if died:
            r = -150

        if won:
            r = 500

        print("Reward: "+str(r))
        return [r]

    def cache(self, old_state, new_state, action, reward):
        self.mem.append(StateStep(old_state, new_state, action, reward))

    def sample_cache(self):
        if len(self.mem) > BATCH_SIZE:
            return sample(self.mem, BATCH_SIZE)
        else:
            return self.mem

    def set_direction(self, relative_direction: int):
        ## direction is a int between 0 and 3
        ## 0 = turn left
        ## 1 = turn right
        ## 2 = go straight
        ## 3 = turn around
        # Convert the relative direction to an absolute direction

        current_direction = self.game.pacman.want_direction
        directions = [LEFT, UP, RIGHT, DOWN]
        if current_direction == STOP:
            current_direction = LEFT

        if current_direction not in directions:
            raise ValueError("Invalid current direction")

        if relative_direction < 0 or relative_direction > 3:
            raise ValueError("Invalid relative direction")

        # Calculate the new index based on the relative direction
        new_index = (directions.index(current_direction) + relative_direction) % 4

        # Return the new absolute direction
        self.game.pacman.want_direction = directions[new_index]
        return directions[new_index]

    def run(self):
        self.game.startGame()
        self.game.ghosts.ghosts = [self.game.ghosts.ghosts[0]]

        self.game.update()

        stuck_steps = 0
        last_position_x = 0
        last_position_y = 0
        self.game.pause.setPause(playerPaused=True)
        episode = 0
        while True:
            self.last_pac_pos = self.game.pacman.position
            self.last_pallets_eaten = 0
            episode += 1
            # Episode
            while True:
                # x und y sind die Koordinaten des Tiles auf der Map
                # Ein tile ist 16x16 Pixel groß und die Map hat ein Offset von 4 Pixeln nach rechts
                x = round((self.game.pacman.position.x + 4) / 16)
                y = round(self.game.pacman.position.y / 16)
                died, won = False, False
                if last_position_x != x or last_position_y != y:
                    # Pacman befindet sich auf einem neuen Tile
                    died, won = self.action_step(self.eval_state())
                    self.stuck = False

                else:
                    stuck_steps += 1
                    self.game.update()

                    if stuck_steps > 4:
                        # Pacman hat nach in Richtung des Richtung der Wand bewegt
                        # Wir müssen die Richtung ändern
                        died, won = self.action_step(self.eval_state())

                        stuck_steps = 0

                last_position_x = x
                last_position_y = y
                if died or won:
                    break
            print("episode finished")
            # Episode beendet -> erneutes Trainieren
            self.train_long()

            if episode%100==0:
                os.makedirs("models", exist_ok=True)
                self.dqn.save(os.path.join("models", "episode_"+str(episode)+".pth"))




if __name__ == '__main__':
    pacman_ki = PacmanKI()
    pacman_ki.run()
