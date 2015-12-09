# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 12:11:17 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import audioFeatureExtractionFunctions as featureExtract
import scipy


filename = "Final_Set5_Norm_Originals/norm_Applause_01_big.wav"
f, sr = librosa.load(filename)


##calculates the Time Homogineity of the function f given a base window size
##func accepts a time searies and retuns a number.
def calculateTimeHomogeneity(filename, func, windowSize):
    f, sr = librosa.load(filename)

    fArray = []
    for i in range(f.size - windowSize):
        fArray.append(func(f[i:i+windowSize]))
    
    funcTotal = func(f)
    
    funcVar = 0
    for i in fArray:
        funcVar += (i-funcTotal)**2
    funcVar /= len(fArray)

    return funcVar

#print calculateTimeHomogeneity(filename, np.mean, f.size/100)
#print calculateTimeHomogeneity(filename, featureExtract.ExtractMelSpectraSparcityFeatures, f.size/100)

def calculateRMSETimeHomogeneity(filename):
    f, sr = librosa.load(filename)
    
    rmse = librosa.feature.rmse(f)
    
    rmseAve = np.mean(rmse)
    
    rmseVar = 0
    for i in rmse[0]:
        #print i
        rmseVar += (i-rmseAve)**2
    rmseVar /= len(rmse)
    
    return rmseVar
    
#print calculateRMSETimeHomogeneity(filename)

    

#print calculateSpectraVarianceTimeHomogienity(filename, librosa.cqt, 10)


def calculateKOrderStatistic(filename, k):
    f, sr = librosa.load(filename)
    return scipy.stats.kstat(f,k)
    
def calculateVarKOrderStatistic(filename, k):
    f, sr = librosa.load(filename)
    return scipy.stats.kstatvar(f,k)

print calculateKOrderStatistic(filename, 1)
print calculateVarKOrderStatistic(filename, 1)
print calculateKOrderStatistic(filename, 2)
print calculateVarKOrderStatistic(filename, 2)
print calculateKOrderStatistic(filename, 3)
print calculateVarKOrderStatistic(filename, 3)
print calculateKOrderStatistic(filename, 4)
print calculateVarKOrderStatistic(filename, 4)

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








#def calculateSpectraAverageTimeHomogineity(filename, spectraTransform, windowSize):
#    f, sr = librosa.load(filename)
#
#    fSpectra = spectraTransform(f)
#
#    fSpectraAve = []
#    fArray = []
#    for i in range(len(fSpectra)):
#        fSpectraAve.append(np.mean(fSpectra[i]))
#        fArray.append([])
#        for j in range(fSpectra[i].size - windowSize):
#            fArray[i].append(np.mean(fSpectra[i][j:j+windowSize]))
#    #print "fSpectraAve ", fSpectraAve
#    print len(fSpectraAve)
#    print fSpectra.shape
#    fSpectraVar = []
#    for i in range(len(fSpectraAve)):
#        fSpectraVar.append(0)
#        for j in fSpectra[i]:
#            fSpectraVar[i] += (j-fSpectraAve[i])**2
#        fSpectraVar[i] /= len(fSpectra[i])
#
#    print len(fSpectraVar)
#    return fSpectraVar
#
##print calculateSpectraAverageTimeHomogineity(filename, librosa.stft, 10)
##print calculateSpectraAverageTimeHomogineity(filename, librosa.cqt, 10)
##print calculateSpectraAverageTimeHomogineity(filename, librosa.feature.melspectrogram, 10)
#
#def calculateSpectraStatisticTimeHomogineity(filename, spectraTransform, statistic, windowSize):
#    f, sr = librosa.load(filename)
#
#    fSpectra = spectraTransform(f)
#
#    fSpectraAve = []
#    fArray = []
#    for i in range(len(fSpectra)):
#        fSpectraAve.append(statistic(fSpectra[i]))
#        fArray.append([])
#        for j in range(fSpectra[i].size - windowSize):
#            fArray[i].append(statistic(fSpectra[i][j:j+windowSize]))
#    #print "fSpectraAve ", fSpectraAve
#    print len(fSpectraAve)
#    print fSpectra.shape
#    fSpectraVar = []
#    for i in range(len(fSpectraAve)):
#        fSpectraVar.append(0)
#        for j in fSpectra[i]:
#            fSpectraVar[i] += (j-fSpectraAve[i])**2
#        fSpectraVar[i] /= len(fSpectra[i])
#
#    print len(fSpectraVar)
#    return fSpectraVar
#    
#def calculateSpectraVarianceTimeHomogienity(filename, spectraTransform, windowSize):
#    return calculateSpectraStatisticTimeHomogineity(filename, spectraTransform, np.var, windowSize)   
#    
##print calculateSpectraVarianceTimeHomogienity(filename, librosa.cqt, 10)
#
#def calculateSpectraSkewTimeHomogienity(filename, spectraTransform, windowSize):
#    return calculateSpectraStatisticTimeHomogineity(filename, spectraTransform, scipy.stats.skews, windowSize)   
#    
##print calculateSpectraVarianceTimeHomogienity(filename, librosa.cqt, 10)
#
#def calculateSpectraKurtosisTimeHomogienity(filename, spectraTransform, windowSize):
#    return calculateSpectraStatisticTimeHomogineity(filename, spectraTransform, scipy.stats.kurtosis, windowSize)   
#    