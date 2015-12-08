# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 12:11:17 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

f, sr = librosa.load("Final_Set5_Norm_Originals/norm_Applause_01_big.wav")

rmsenergy = librosa.feature.rmse(y=f)
melspectra = librosa.feature.melspectrogram(f)
frequencyDomain = librosa.stft(f)
constQ = librosa.cqt(f)

print len(rmsenergy[0])

plt.plot(rmsenergy[0])
plt.show()


print len(melspectra)
print len(melspectra[0])

epsilon = .1

melEpsilonSparcityMatrix = (melspectra < epsilon)

melSparcity = melspectra.size - np.count_nonzero(melspectra)

melEpsilonSparcity = float((melEpsilonSparcityMatrix.size - np.count_nonzero(melEpsilonSparcityMatrix)))/melEpsilonSparcityMatrix.size

print "size", melspectra.size
print "Melspectra epsilon sparcity for epsilon = ", epsilon, " is ", melEpsilonSparcity

melSpectraMin = np.amin(melSparcity)

print "Minimum Melspectra element", melSpectraMin

print "melspectra number of zero entries", melSparcity

melave = np.mean(melspectra, axis=0)
melSparcityTimeAve = len(melave) - np.count_nonzero(melave)
print "len of melave = ", len(melave)
print "number of nonzero entries of the melspectra average values", melSparcityTimeAve



#plt.specgram(melspectra, NFFT = 302)
plt.pcolormesh(melspectra)
#plt.show()

plt.specgram(melave)
#plt.show()

print len(frequencyDomain)
print len(frequencyDomain[0])

#plt.specgram(frequencyDomain, NFFT = 302)
plt.pcolormesh(frequencyDomain)
#plt.show()

print len(f)