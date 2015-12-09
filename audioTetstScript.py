# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 12:11:17 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

f, sr = librosa.load("Final_Set5_Norm_Originals/norm_Applause_01_big.wav")

def calculateTimeHomogeneity(filename, f, windowSize):
    




#rmsenergy = librosa.feature.rmse(y=f)
#melspectra = librosa.feature.melspectrogram(f)
#frequencyDomain = librosa.stft(f)
#constQ = librosa.cqt(f)
#
#print len(rmsenergy[0])
#
#plt.plot(rmsenergy[0])
#plt.show()
#
#
#print len(melspectra)
#print len(melspectra[0])
#
#epsilon = .0001
#
#
###### Sparcity - fraction of entries that are zero (or within \epsilon)
###### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
#
#melEpsilonSparcityMatrix = (melspectra > epsilon)
#melSparcity = melspectra.size - np.count_nonzero(melspectra)
#melEpsilonSparcity = float((melEpsilonSparcityMatrix.size - np.count_nonzero(melEpsilonSparcityMatrix)))/melEpsilonSparcityMatrix.size
#
#melSpectraMin = np.amin(melSparcity)
#
#print "size", melspectra.size
#print "Minimum Melspectra element", melSpectraMin
#print "melspectra number of zero entries", melSparcity
#print "Melspectra epsilon sparcity for epsilon = ", epsilon, " is ", melEpsilonSparcity
#
#melSpectraMax = np.amax(melspectra, axis=1)
#melSpectraBandSparcityMatrix = (melSpectraMax > epsilon)
#melBandSparcity = float(len(melSpectraMax) - np.count_nonzero(melSpectraBandSparcityMatrix))/len(melSpectraMax)
#
#print "Band sparcity for mel spectra: ", melBandSparcity
#
#
#melave = np.mean(melspectra, axis=1)
#melAveSpectraBandSparcityMatrix = (melave > epsilon)
#melSparcityTimeAve = float(len(melave) - np.count_nonzero(melAveSpectraBandSparcityMatrix))/len(melave)
#
#
#print "Band sparcity based on ave mel spectra: ", melSparcityTimeAve
#
#print "Mel spectra: "
##plt.specgram(melspectra, NFFT = 302)
#plt.pcolormesh(melspectra)
#plt.show()
#
#print "Average Spectra: "
#plt.specgram(melave)
#plt.show()
#
#
#
#
#
#
#
#
#
#print len(frequencyDomain)
#print len(frequencyDomain[0])

#plt.specgram(frequencyDomain, NFFT = 302)
##plt.show()

#print len(f)