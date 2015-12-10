from audioFeatureExtractionFunctions import *
from common import base_names, ys, sound_path
from compression import compression_rate
import librosa
import numpy as np
import scipy
import cProfile
from sklearn import linear_model

spectra = [librosa.cqt, librosa.stft, librosa.feature.melspectrogram]
moments = [np.var, scipy.stats.skew, scipy.stats.kurtosis]

def feature_vector(base_name):
    filename = sound_path(base_name)

    fv = np.array([])
    fv = np.append(fv, calculateRMSE(filename))
    fv = np.append(fv, compression_rate(base_name))

    fv = np.append(fv, ExtractTemporalSparcity(filename))
    fv = np.append(fv, ExtractMelSpectraSparcityFeatures(filename))
    fv = np.append(fv, ExtractCQSpectraSparcityFeatures(filename))
    fv = np.append(fv, ExtractSTFTSpectraSparcityFeatures(filename))
    fv = np.append(fv, calculateRMSETimeHomogeneity(filename))

    """
    for spectrum in spectra:
        for moment in moments:
            fv = np.append(fv, calculateSpectraStatisticTimeHomogeneity(filename, spectrum, moment, 10))
    """
    fv = np.append(fv, calculateCrossCorrelations(filename, spectra[0]))
#    fv = np.append(fv, twoLayerTransform(filename, spectra[0]))
#    fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(filename, spectra[0], np.mean, 10))
#    fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(filename, spectra[0], np.var, 10))
    
    return fv

def lasso(training, validation, alpha):
    model = linear_model.Lasso(alpha=alpha)
    model.fit(training[:,:-1], training[:, -1])
    print model.score(validation[:,:-1], validation[:, -1])
    return model


fvs = [feature_vector(base_name) for base_name in base_names]
tr_len = len(fvs)/2
val_len = len(fvs)/4

fvs = np.c_[fvs, ys]

fvs = np.random.permutation(fvs)

alphas = [10** i for i in range(-8,4)]
models = [lasso(fvs[:tr_len], fvs[tr_len:tr_len + val_len], alpha) for alpha in alphas]
