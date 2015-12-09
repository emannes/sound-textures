# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 16:58:31 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

defaultEpsilon = 10**-4

def ExtractTemporalSparcity(filename, epsilon=defaultEpsilon):
    
    f, sr = librosa.load(filename)
    
    timeEpsilonSparcityMatrix = (f > epsilon)
    timeEpsilonSparcity = float(f.size-np.count_nonzero(timeEpsilonSparcityMatrix))/f.size
    
        
    
    return timeEpsilonSparcity

def ExtractMelSpectraSparcityFeatures(filename, epsilon=defaultEpsilon):
    
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

def ExtractCQSpectraSparcityFeatures(filename, epsilon=defaultEpsilon):
    
    f, sr = librosa.load(filename)    
    
    cqtspectra = librosa.cqt(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    cqtEpsilonSparcityMatrix = (cqtspectra > epsilon)
    cqtSparcity = cqtspectra.size - np.count_nonzero(cqtspectra)
    cqtEpsilonSparcity = float((cqtEpsilonSparcityMatrix.size - np.count_nonzero(cqtEpsilonSparcityMatrix)))/cqtEpsilonSparcityMatrix.size
    
    print "size", cqtspectra.size
    print "cqtspectra epsilon sparcity for epsilon = ", epsilon, " is ", cqtEpsilonSparcity
    
    cqtSpectraMax = np.amax(cqtspectra, axis=1)
    cqtSpectraBandSparcityMatrix = (cqtSpectraMax > epsilon)
    cqtBandSparcity = float(len(cqtSpectraMax) - np.count_nonzero(cqtSpectraBandSparcityMatrix))/len(cqtSpectraMax)
    
    print "Epsilon Band sparcity for cqt spectra: ", cqtBandSparcity
    
    
    cqtave = np.mean(cqtspectra, axis=1)
    cqtAveSpectraBandSparcityMatrix = (cqtave > epsilon)
    cqtBandSparcityTimeAve = float(len(cqtave) - np.count_nonzero(cqtAveSpectraBandSparcityMatrix))/len(cqtave)
    
    
    print "Epsilon Band sparcity based on ave cqt spectra: ", cqtBandSparcityTimeAve
    
    return cqtEpsilonSparcity, cqtBandSparcity, cqtBandSparcityTimeAve
    
    
def ExtractSTFTSpectraSparcityFeatures(filename, epsilon=defaultEpsilon):
    
    f, sr = librosa.load(filename)    
    
    stftspectra = librosa.stft(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    stftEpsilonSparcityMatrix = (stftspectra > epsilon)
    stftSparcity = stftspectra.size - np.count_nonzero(stftspectra)
    stftEpsilonSparcity = float((stftEpsilonSparcityMatrix.size - np.count_nonzero(stftEpsilonSparcityMatrix)))/stftEpsilonSparcityMatrix.size
    
    print "size", stftspectra.size
    print "stftspectra epsilon sparcity for epsilon = ", epsilon, " is ", stftEpsilonSparcity
    
    stftSpectraMax = np.amax(stftspectra, axis=1)
    stftSpectraBandSparcityMatrix = (stftSpectraMax > epsilon)
    stftBandSparcity = float(len(stftSpectraMax) - np.count_nonzero(stftSpectraBandSparcityMatrix))/len(stftSpectraMax)
    
    print "Epsilon Band sparcity for stft spectra: ", stftBandSparcity
    
    
    stftave = np.mean(stftspectra, axis=1)
    stftAveSpectraBandSparcityMatrix = (stftave > epsilon)
    stftBandSparcityTimeAve = float(len(stftave) - np.count_nonzero(stftAveSpectraBandSparcityMatrix))/len(stftave)
    
    
    print "Epsilon Band sparcity based on ave stft spectra: ", stftBandSparcityTimeAve
    
    return stftEpsilonSparcity, stftBandSparcity, stftBandSparcityTimeAve

