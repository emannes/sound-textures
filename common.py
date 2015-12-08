import os

SOUND_DIR = "Final_Set5_Norm_Originals/"
COMPRESSED_DIR = "compressed_sounds/"

sound_files = [f for f in os.listdir(SOUND_DIR) if os.path.isfile(os.path.join(SOUND_DIR, f))]
base_names = [f[:-4] for f in sound_files]

def sound_path(base_name):
    return os.path.join(SOUND_DIR, base_name + ".wav")

def compressed_path(base_name):
    return os.path.join(COMPRESSED_DIR, base_name + ".ogg")
