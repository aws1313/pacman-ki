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

MEM_SIZE = 1000000000
BATCH_SIZE = 70000
LR = 0.002
GAMMA = 0.9
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

# Definiere die Konstanten der Epsilon Funktion
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = .99999

torch.cuda.set_device(0)

StateStep = namedtuple("StateStep", ["old_state", "new_state", "action", "reward"])


class PacmanKI:
    def __init__(self):
        # Initialisiere das Pac-Man Spiel
        self.game = GameController()

        # Initialisiere den DQN Algorithmus
        self.mem = deque(maxlen=MEM_SIZE)
        self.dqn = DQN(981, 32, 4, LR, GAMMA, DEVICE).to(DEVICE)

        # Initialisiere verschiedene Variablen für das Training
        self.game_count = 0
        self.epsilon = 0
        self.steps_done = 0
        self.last_pallets_eaten = 0
        self.last_pac_pos = None
        self.action_count = 0
        self.mode_max = True

    def train_long(self):
        cache = self.sample_cache()
        old_states, new_states, actions, rewards = zip(*cache)

        self.dqn.train_step(np.array(old_states), np.array(new_states), actions, rewards)

    def action_step(self, old_state):
        # Wähle, ob die Aktion zufällig oder durch das Modell bestimmt wird

        global EPS_START
        print("rand prob: "+str(EPS_START))
        if random() <= EPS_START:
            # Zufällige Aktion
            action = randint(0, 3)
        else:
            # Aktion durch das Modell bestimmt
            action = torch.argmax(self.dqn(torch.from_numpy(old_state).to(DEVICE))).item()

        EPS_START *= EPS_DECAY
        EPS_START = max(EPS_START, EPS_END)

        self.action_count += 1
        self.steps_done += 1

        # Führe die Aktion aus und speichere den Zustand
        action = self.set_direction(action)
        new_state, reward, died, won = self.step()
        self.cache(old_state, new_state, action, reward)
        self.dqn.train_step(old_state, new_state, action, reward)
        return died, won

    def step(self, ):
        # Führe einen Schritt im Spiel aus
        self.game.update()
        died = False
        if self.game.pacman_died:
            self.game.pacman_died = False
            died = True
        # new_state, reward, died, won
        return self.eval_state(), self.eval_reward(died, False), died, False

    def tile_from_pos(self, x, y):
        # Konvertiere die Pixelkoordinaten in die Tile-Koordinaten
        x = round((x + 4) / 16)
        y = round(y / 16)
        return x, y

    def eval_state(self):
        if self.mode_max:
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
        pellets.extend(self.game.pallets_eaten)
        pellets.extend(self.game.pellets.pelletList)
        pellets.sort(key=operator.attrgetter('id'))
        ps = []
        for p in pellets:
            sup = isinstance(p, PowerPellet)
            ps.extend([p.position.x, p.position.y, sup, p.eaten])

        ghosts = []
        for g in self.game.ghosts.ghosts:
            ghosts.extend([g.position.x, g.position.y])

        s = [
            self.game.pacman.want_direction,
            self.game.pacman.position.x,
            self.game.pacman.position.y
        ]
        s.extend(ps)
        s.extend(ghosts)
        return np.array(s, dtype=np.float32)

    def vec2_to_tp(self, vec):
        return [vec.x, vec.y]

    def eval_reward(self, died, won):
        pl = (self.game.pellets.numEaten - self.last_pallets_eaten) * 10
        self.last_pallets_eaten = self.game.pellets.numEaten
        # s = 0
        s = -3 if self.last_pac_pos == self.game.pacman.position else 0

        self.last_pac_pos = self.game.pacman.position

        r = pl + s + 1

        if died:
            r = -2000

        if won:
            r = 4000

        print("Reward: " + str(r))
        return [r]

    def cache(self, old_state, new_state, action, reward):
        self.mem.append(StateStep(old_state, new_state, action, reward))

    def sample_cache(self):
        if len(self.mem) > BATCH_SIZE:
            return sample(self.mem, BATCH_SIZE)
        else:
            return self.mem

    def set_direction(self, relative_direction: int):
        # Konvertiere die absolute Richtung in eine relative Richtung

        current_direction = self.game.pacman.want_direction
        directions = [LEFT, UP, RIGHT, DOWN]
        if current_direction == STOP:
            current_direction = LEFT

        if current_direction not in directions:
            raise ValueError("Invalid current direction")

        if relative_direction < 0 or relative_direction > 3:
            raise ValueError("Invalid relative direction")

        new_index = (directions.index(current_direction) + relative_direction) % 4

        # Setze die neue Richtung im Spiel
        self.game.pacman.want_direction = directions[new_index]
        return directions[new_index]

    def run(self):
        # Starte das Spiel und trainiere das Modell
        self.game.startGame()
        self.game.ghosts.ghosts = [self.game.ghosts.ghosts[0]]

        self.game.update()
        self.mode_max = False
        self.game.pause.setPause(playerPaused=True)
        episode = 0
        while True:
            # Starte eine neue Episode
            self.last_pac_pos = self.game.pacman.position
            self.last_pallets_eaten = 0
            self.action_count = 0
            episode += 1
            died, won = False, False

            stuck_steps = 0
            last_position_x = 0
            last_position_y = 0
            while not (died or won):
                # Führe Schritte aus, bis das Spiel beendet ist
                x = round((self.game.pacman.position.x + 4) / 16)
                y = round(self.game.pacman.position.y / 16)

                if last_position_x != x or last_position_y != y:
                    # Führe einen Schritt aus, sobald sich die Position geändert hat
                    died, won = self.action_step(self.eval_state())
                    self.stuck = False

                elif not (died or won):
                    stuck_steps += 1
                    self.game.update()

                    if stuck_steps > 4:
                        # Pacman hat nach in Richtung der Wand bewegt
                        # -> Er muss die richtung ändern
                        died, won = self.action_step(self.eval_state())

                        stuck_steps = 0

                last_position_x = x
                last_position_y = y
            print("episode finished with {} actions".format(self.action_count))
            # Episode beendet -> erneutes Trainieren
            if episode % 50 == 0:
                self.train_long()

            # Speichere das Modell alle 100 Episoden ab
            if episode % 100 == 0:
                os.makedirs("models", exist_ok=True)
                self.dqn.save(os.path.join("models", "episode_" + str(episode) + ".pth"))


if __name__ == '__main__':
    pacman_ki = PacmanKI()
    pacman_ki.run()
