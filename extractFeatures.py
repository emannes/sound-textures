from audioFeatureExtractionFunctions import *
from common import base_names, ys
from common import sound_path
#from common import normalized_path as sound_path
from compression import compression_rate
import librosa
import numpy as np
import scipy
import cProfile
from sklearn import linear_model
from sklearn import preprocessing
import cPickle as pickle

MAX_FEATURES = 15

spectra = [librosa.cqt, librosa.stft, librosa.feature.melspectrogram]
moments = [np.var, scipy.stats.skew, scipy.stats.kurtosis]

def feature_vector(base_name):
    print base_name
    f = sound_path(base_name)
    f, sr = librosa.load(f)

    fv = np.array([])
    fv = np.append(fv, calculateRMSE(f))
    fv = np.append(fv, compression_rate(base_name))

    fv = np.append(fv, ExtractTemporalSparcity(f))
    fv = np.append(fv, ExtractMelSpectraSparcityFeatures(f))
    fv = np.append(fv, ExtractCQSpectraSparcityFeatures(f))
    fv = np.append(fv, ExtractSTFTSpectraSparcityFeatures(f))
    fv = np.append(fv, calculateRMSETimeHomogeneity(f))

    for moment in moments:
        fv = np.append(fv, calculateSpectraStatisticTimeHomogeneity(f, librosa.cqt, moment, 10))

    """
    for spectrum in spectra:
        for moment in moments:
            fv = np.append(fv, calculateSpectraStatisticTimeHomogeneity(f, spectrum, moment, 10))
    """
    fv = np.append(fv, calculateCrossCorrelations(f, spectra[0]))
    fv = np.append(fv, twoLayerTransform(f, spectra[0]))
    fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(f, spectra[0], np.mean, 10))
    fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(f, spectra[0], np.var, 10))
    
    return fv

def lasso(training, validation, alpha):
    model = linear_model.Lasso(alpha=alpha)
    model.fit(training[:,:MAX_FEATURES], training[:, -1])
    print model.score(validation[:,:MAX_FEATURES], validation[:, -1])
    return model

fvs = [feature_vector(base_name) for base_name in base_names]
#fvs = preprocessing.scale(fvs, axis=1)

goodlength = max([len(row) for row in fvs])
goodis = [i for i in range(len(row)) if len(fvs[i]) == goodlength]
fvs2 = np.array([fvs[i] for i in goodis])
goodys = [ys[i] for i in goodis]

fvs3 = np.c_[fvs2, goodys]


fvsfile = open('fvs2.pkl','w')
pickle.dump([fvs, goodis, fvs3], fvsfille)
fvsfile.close()

"""
fvsfile = open('fvs.pkl','r')
fvs = pickle.load(fvsfile)
fvsfile.close()
"""
fvs = fvs3

tr_len = len(fvs)/2
val_len = len(fvs)/4

fvs = np.random.permutation(fvs)

alphas = [10** i for i in range(-8,10)]
models = [lasso(fvs[:tr_len], fvs[tr_len:tr_len + val_len], alpha) for alpha in alphas]
