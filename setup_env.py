from zipfile import ZipFile
import wget
import shutil
import os
import pc_constants as c

if __name__ == '__main__':
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
        print("Pacman wurde heruntergeladen. Bitte erneut öffnen")
        exit()
    print(os.getcwd())