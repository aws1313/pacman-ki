import pc_constants as c
import sys
sys.path.append(c.GAME_DIR)
import os
from zipfile import ZipFile
import wget
import shutil
import time

def setup_environment():
    if not os.path.exists(c.SETUP_DONE_FILE):
        os.makedirs(c.GAME_DIR, exist_ok=True)
        wget.download(c.GAME_URL, out=c.GAME_ZIP_FILE_NAME)
        with ZipFile(c.GAME_ZIP_FILE_NAME, "r") as z:
            z.extractall(c.GAME_DIR)
        for filename in os.listdir(os.path.join(c.GAME_DIR, "Pacman_Complete")):
            shutil.move(f"{c.GAME_DIR}/Pacman_Complete/{filename}", c.GAME_DIR)
        os.removedirs(os.path.join(c.GAME_DIR, "Pacman_Complete"))
        os.remove(c.GAME_ZIP_FILE_NAME)
        with open(c.SETUP_DONE_FILE, "w") as f:
            f.write("ok")
    os.chdir(c.GAME_DIR)
    print(os.getcwd())


setup_environment()


from game.run import *


class PacmanKI:
    def __init__(self):
        pass

    def run(self):
        tgame = GameController()
        tgame.startGame()
        tgame.ghosts.ghosts.clear()
        post = tgame.pacman.target
        pacpos = tgame.pacman.position

        while True:
            if post.position != tgame.pacman.target.position and tgame.pacman.position == post.position and pacpos != tgame.pacman.position:
                print("change "+str(time.time()))
                tgame.pacman.direction = STOP
                tgame.pacman.target = post
                pacpos = tgame.pacman.position
            post = tgame.pacman.target
            if tgame.pacman.direction == STOP:
                print(tgame.pacman.node.neighbors)
            tgame.update()




if __name__ == '__main__':
    pacman_ki = PacmanKI()
    pacman_ki.run()
