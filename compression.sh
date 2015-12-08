#!/bin/bash

FILES=Final_Set5_Norm_Originals/*
COMPRESSEDDIR=compressed_sounds/
for f in $FILES
do
    echo "Compressing $f"
    ffmpeg -i "$f" -c:a libvorbis "$COMPRESSEDDIR`basename "$f" .wav`.ogg"
done
