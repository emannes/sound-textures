from audioFeatureExtractionFunctions import *
from common import base_names, ys, sound_path
from compression import compression_rate
import librosa
import numpy as np
import scipy
import cProfile

spectra = [librosa.cqt, librosa.stft, librosa.feature.melspectrogram]
moments = [np.var, scipy.stats.skew, scipy.stats.kurtosis]

def feature_vector(base_name):
    filename = sound_path(base_name)

    fv = np.array([])
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
    #spectrastatistictimehomogeneity TK
    #crosscorrelations TK
    
fvs = [feature_vector(base_name) for base_name in base_names]
