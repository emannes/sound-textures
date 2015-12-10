import os
import csv

SOUND_DIR = "Final_Set5_Norm_Originals/"
COMPRESSED_DIR = "compressed_sounds/"
NORMALIZED_DIR = "normalized/"
SCORES = "scores/scores-certain.csv"

def sound_path(base_name):
    return os.path.join(SOUND_DIR, base_name + ".wav")

def compressed_path(base_name):
    return os.path.join(COMPRESSED_DIR, base_name + ".ogg")



sound_files = [f for f in os.listdir(SOUND_DIR) if os.path.isfile(os.path.join(SOUND_DIR, f))]
base_names_all = [f[:-4] for f in sound_files]

with open(SCORES, 'r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

base_names = [row[0] for row in rows]
ys = [float(row[1]) for row in rows]

