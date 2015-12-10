import os
import csv
import librosa

SOUND_DIR = "Final_Set5_Norm_Originals/"
COMPRESSED_DIR = "compressed_sounds/"
NORMALIZED_DIR = "normalized/"
SCORES = "scores/scores-certain.csv"
BLACKLIST = [] #["norm_CSE-23 Newspaper printing press", "norm_SE2-25 Horse Trot On Cobblestones", "norm_SE3-02 Brushing Teeth", "norm_writing-pen_on_paper2"]

def sound_path(base_name):
    return os.path.join(SOUND_DIR, base_name + ".wav")

def compressed_path(base_name):
    return os.path.join(COMPRESSED_DIR, base_name + ".ogg")

def normalized_path(base_name):
    return os.path.join(NORMALIZED_DIR, base_name + ".wav")

def normalize(base_name):
    s = librosa.load(sound_path(base_name))
    librosa.output.write_wav(normalized_path(base_name), s[0], s[1])

sound_files = [f for f in os.listdir(SOUND_DIR) if os.path.isfile(os.path.join(SOUND_DIR, f))]
base_names_all = [f[:-4] for f in sound_files]

with open(SCORES, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

base_names = [row[0] for row in rows if row[0] not in BLACKLIST]l
ys = [float(row[1]) for row in rows if row[0] not in BLACKLIST]
    
def normalize_all():
    for base_name in base_names_all:
        normalize(base_name)




