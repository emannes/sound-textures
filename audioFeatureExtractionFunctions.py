# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 16:58:31 2015

@author: Jayson
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy

defaultEpsilon = 10**-4
defaultBandNumber1 = 6
defaultBandNumber2 = 30

def ExtractTemporalSparcity(f, epsilon=defaultEpsilon):
    
    timeEpsilonSparcityMatrix = (f > epsilon)
    timeEpsilonSparcity = float(f.size-np.count_nonzero(timeEpsilonSparcityMatrix))/f.size
    
    return timeEpsilonSparcity

def ExtractMelSpectraSparcityFeatures(f, epsilon=defaultEpsilon):
    
    melspectra = librosa.feature.melspectrogram(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    melEpsilonSparcityMatrix = (melspectra > epsilon)
    melSparcity = melspectra.size - np.count_nonzero(melspectra)
    melEpsilonSparcity = float((melEpsilonSparcityMatrix.size - np.count_nonzero(melEpsilonSparcityMatrix)))/melEpsilonSparcityMatrix.size
    
    #print "size", melspectra.size
    #print "Melspectra epsilon sparcity for epsilon = ", epsilon, " is ", melEpsilonSparcity
    
    melSpectraMax = np.amax(melspectra, axis=1)
    melSpectraBandSparcityMatrix = (melSpectraMax > epsilon)
    melBandSparcity = float(len(melSpectraMax) - np.count_nonzero(melSpectraBandSparcityMatrix))/len(melSpectraMax)
    
    #print "Epsilon Band sparcity for mel spectra: ", melBandSparcity
    
    
    melave = np.mean(melspectra, axis=1)
    melAveSpectraBandSparcityMatrix = (melave > epsilon)
    melBandSparcityTimeAve = float(len(melave) - np.count_nonzero(melAveSpectraBandSparcityMatrix))/len(melave)
    
    
    #print "Epsilon Band sparcity based on ave mel spectra: ", melBandSparcityTimeAve
    
    return melEpsilonSparcity, melBandSparcity, melBandSparcityTimeAve

def ExtractCQSpectraSparcityFeatures(f, epsilon=defaultEpsilon):
    
    cqtspectra = librosa.cqt(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    cqtEpsilonSparcityMatrix = (cqtspectra > epsilon)
    #cqtSparcity = cqtspectra.size - np.count_nonzero(cqtspectra)
    cqtEpsilonSparcity = float((cqtEpsilonSparcityMatrix.size - np.count_nonzero(cqtEpsilonSparcityMatrix)))/cqtEpsilonSparcityMatrix.size
    
    #print "size", cqtspectra.size
    #print "cqtspectra epsilon sparcity for epsilon = ", epsilon, " is ", cqtEpsilonSparcity
    
    cqtSpectraMax = np.amax(cqtspectra, axis=1)
    cqtSpectraBandSparcityMatrix = (cqtSpectraMax > epsilon)
    cqtBandSparcity = float(len(cqtSpectraMax) - np.count_nonzero(cqtSpectraBandSparcityMatrix))/len(cqtSpectraMax)
    
    #print "Epsilon Band sparcity for cqt spectra: ", cqtBandSparcity
    
    
    cqtave = np.mean(cqtspectra, axis=1)
    cqtAveSpectraBandSparcityMatrix = (cqtave > epsilon)
    cqtBandSparcityTimeAve = float(len(cqtave) - np.count_nonzero(cqtAveSpectraBandSparcityMatrix))/len(cqtave)
    
    
    #print "Epsilon Band sparcity based on ave cqt spectra: ", cqtBandSparcityTimeAve
    
    return cqtEpsilonSparcity, cqtBandSparcity, cqtBandSparcityTimeAve
    
    
def ExtractSTFTSpectraSparcityFeatures(f, epsilon=defaultEpsilon):
    
    stftspectra = librosa.stft(f)   
    
    ##### Sparcity - fraction of entries that are zero (or within \epsilon)
    ##### We can have Sparcity of a full spectra or the max over time (if it is ever spase in that band).
    
    stftEpsilonSparcityMatrix = (stftspectra > epsilon)
    #stftSparcity = stftspectra.size - np.count_nonzero(stftspectra)
    stftEpsilonSparcity = float((stftEpsilonSparcityMatrix.size - np.count_nonzero(stftEpsilonSparcityMatrix)))/stftEpsilonSparcityMatrix.size
    
    #print "size", stftspectra.size
    #print "stftspectra epsilon sparcity for epsilon = ", epsilon, " is ", stftEpsilonSparcity
    
    stftSpectraMax = np.amax(stftspectra, axis=1)
    stftSpectraBandSparcityMatrix = (stftSpectraMax > epsilon)
    stftBandSparcity = float(len(stftSpectraMax) - np.count_nonzero(stftSpectraBandSparcityMatrix))/len(stftSpectraMax)
    
    #print "Epsilon Band sparcity for stft spectra: ", stftBandSparcity
    
    
    stftave = np.mean(stftspectra, axis=1)
    stftAveSpectraBandSparcityMatrix = (stftave > epsilon)
    stftBandSparcityTimeAve = float(len(stftave) - np.count_nonzero(stftAveSpectraBandSparcityMatrix))/len(stftave)
    
    
    #print "Epsilon Band sparcity based on ave stft spectra: ", stftBandSparcityTimeAve
    
    return stftEpsilonSparcity, stftBandSparcity, stftBandSparcityTimeAve

def calculateRMSETimeHomogeneity(f):
    
    rmse = librosa.feature.rmse(f)
    
    rmseAve = np.mean(rmse)
    
    rmseVar = 0
    for i in rmse[0]:
        #print i
        rmseVar += (i-rmseAve)**2
    rmseVar /= len(rmse)
    
    return rmseVar

def calculateRMSE(f):
    
    rmse = librosa.feature.rmse(f)
    
    return np.mean(rmse)


def calculateSpectraAverageTimeHomogineity(f, spectraTransform, windowSize):

    fSpectra = spectraTransform(f)

    fSpectraAve = []
    fArray = []
    for i in range(len(fSpectra)):
        fSpectraAve.append(np.mean(fSpectra[i]))
        fArray.append([])
        for j in range(fSpectra[i].size - windowSize):
            fArray[i].append(np.mean(fSpectra[i][j:j+windowSize]))
    #print "fSpectraAve ", fSpectraAve
    #print len(fSpectraAve)
    #print fSpectra.shape
    fSpectraVar = []
    for i in range(len(fSpectraAve)):
        fSpectraVar.append(0)
        for j in fSpectra[i]:
            fSpectraVar[i] += (j-fSpectraAve[i])**2
        fSpectraVar[i] /= len(fSpectra[i])

    #print len(fSpectraVar)
    return fSpectraVar

#print calculateSpectraAverageTimeHomogineity(filename, librosa.stft, 10)
#print calculateSpectraAverageTimeHomogineity(filename, librosa.cqt, 10)
#print calculateSpectraAverageTimeHomogineity(filename, librosa.feature.melspectrogram, 10)

def calculateSpectraStatisticTimeHomogeneity(f, spectraTransform, statistic, windowSize):

    fSpectra = spectraTransform(f)

    fSpectraAve = []
    fArray = []
    for i in range(len(fSpectra)):
        fSpectraAve.append(statistic(fSpectra[i]))
        fArray.append([])
        for j in range(fSpectra[i].size - windowSize):
            fArray[i].append(statistic(fSpectra[i][j:j+windowSize]))
    #print "fSpectraAve ", fSpectraAve
    #print len(fSpectraAve)
    #print fSpectra.shape
    fSpectraVar = []
    for i in range(len(fSpectraAve)):
        fSpectraVar.append(0)
        for j in fSpectra[i]:
            fSpectraVar[i] += (j-fSpectraAve[i])**2
        fSpectraVar[i] /= len(fSpectra[i])

    #print len(fSpectraVar)
    return fSpectraVar
    
def calculateSpectraVarianceTimeHomogeneity(f, spectraTransform, windowSize):
    return calculateSpectraStatisticTimeHomogineity(f, spectraTransform, np.var, windowSize)   
    
#print calculateSpectraVarianceTimeHomogeneity(filename, librosa.cqt, 10)

def calculateSpectraSkewTimeHomogeneity(f, spectraTransform, windowSize):
    return calculateSpectraStatisticTimeHomogineity(f, spectraTransform, scipy.stats.skews, windowSize)   
    
#print calculateSpectraVarianceTimeHomogeneity(filename, librosa.cqt, 10)

def calculateSpectraKurtosisTimeHomogeneity(f, spectraTransform, windowSize):
    return calculateSpectraStatisticTimeHomogineity(f, spectraTransform, scipy.stats.kurtosis, windowSize)   

def calculateCrossCorrelations(f, spectraTransform, n_bins=defaultBandNumber1):
    fSpectra = spectraTransform(f, n_bins=n_bins)

    return np.corrcoef(fSpectra)
    
def twoLayerTransform(f, spectraTransform):
    
    frequencySubbands = spectraTransform(f, n_bins=defaultBandNumber1, hop_length=64)
 
    modulationSubbands = []
    
    for i in range(len(frequencySubbands)):
        modulationEnvelope = np.abs(scipy.signal.hilbert(frequencySubbands[i]))
#        print "modulatin envelope size ", len(modulationEnvelope)
        modulationSubbands.append(spectraTransform(modulationEnvelope, n_bins=defaultBandNumber2, hop_length=64))
        
    return np.array(modulationSubbands)
    
def calculateVarStatisticOfArray(signal, statistic, windowSize):
    signalStatistic = []
    fArray = []
    for i in range(len(signal)):
        signalStatistic.append(statistic(signal[i]))
        fArray.append([])
        for j in range(signal[i].size - windowSize):
            fArray[i].append(statistic(signal[i][j:j+windowSize]))
    #print "fSpectraAve ", fSpectraAve
    #print len(fSpectraAve)
    #print fSpectra.shape
    signalStatisticVar = []
    for i in range(len(signalStatistic)):
        signalStatisticVar.append(0)
        for j in signal[i]:
            signalStatisticVar[i] += (j-signalStatistic[i])**2
        signalStatisticVar[i] /= len(signal[i])

    #print len(fSpectraVar)
    return signalStatisticVar
        

def calculateModulationSubbandKStatisticTimeHomogineity(f, spectraTransform, statistic, windowSize):
    bands = twoLayerTransform(f, spectraTransform)
    
    bandHomogineity = []
    #for i in range(len(bands)):
    for j in bands:
        bandHomogineity.append(calculateVarStatisticOfArray(j, statistic, windowSize))

    bandHomogineity = np.array(bandHomogineity)
    print "bandHomogineity size ", bandHomogineity.shape        
        
    return bandHomogineity
