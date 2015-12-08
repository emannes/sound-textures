# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 16:58:31 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

def ExtractMelSpectraSparcityFeatures(filename, epsilon):
    
    f, sr = librosa.load(filename)    
    
    melspectra = librosa.feature.melspectrogram(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    melEpsilonSparcityMatrix = (melspectra > epsilon)
    melSparcity = melspectra.size - np.count_nonzero(melspectra)
    melEpsilonSparcity = float((melEpsilonSparcityMatrix.size - np.count_nonzero(melEpsilonSparcityMatrix)))/melEpsilonSparcityMatrix.size
    
    print "size", melspectra.size
    print "Melspectra epsilon sparcity for epsilon = ", epsilon, " is ", melEpsilonSparcity
    
    melSpectraMax = np.amax(melspectra, axis=1)
    melSpectraBandSparcityMatrix = (melSpectraMax > epsilon)
    melBandSparcity = float(len(melSpectraMax) - np.count_nonzero(melSpectraBandSparcityMatrix))/len(melSpectraMax)
    
    print "Epsilon Band sparcity for mel spectra: ", melBandSparcity
    
    
    melave = np.mean(melspectra, axis=1)
    melAveSpectraBandSparcityMatrix = (melave > epsilon)
    melBandSparcityTimeAve = float(len(melave) - np.count_nonzero(melAveSpectraBandSparcityMatrix))/len(melave)
    
    
    print "Epsilon Band sparcity based on ave mel spectra: ", melBandSparcityTimeAve
    
    return melEpsilonSparcity, melBandSparcity, melBandSparcityTimeAve
