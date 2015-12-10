from audioFeatureExtractionFunctions import *
from common import base_names, ys
#from common import sound_path
from common import normalized2_path as sound_path
from compression import compression_rate
import librosa
import numpy as np
import scipy
import cProfile
from sklearn import linear_model
from sklearn import preprocessing
import cPickle as pickle
import matplotlib as pyplot

MAX_FEATURES = 100

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


#    for moment in moments:
#        fv = np.append(fv, calculateSpectraStatisticTimeHomogeneity(f, librosa.cqt, moment, 10))


    fv = np.append(fv, calculateCrossCorrelations(f, spectra[0]))
#    fv = np.append(fv, twoLayerTransform(f, spectra[0]))
 #   fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(f, spectra[0], np.mean, 10))
  #  fv = np.append(fv, calculateModulationSubbandKStatisticTimeHomogineity(f, spectra[0], np.var, 10))

    """
    for spectrum in spectra:
        for moment in moments:
            fv = np.append(fv, calculateSpectraStatisticTimeHomogeneity(f, spectrum, moment, 10))
    """
    return fv

def lasso(training, validation, alpha):
    model = linear_model.Lasso(alpha=alpha)
    model.fit(training[:,:-1], training[:, -1])
    print model.score(validation[:,:-1], validation[:, -1])
    return model

def lasso2(training, validation, alpha,i):
    model = linear_model.Lasso(alpha=alpha)
    model.fit(training[:,i:i+1], training[:, -1])
#    print model.score(validation[:,i:i+1], validation[:, -1])
    return model



fvs = [feature_vector(base_name) for base_name in base_names]
print fvs.shape
#fvs = preprocessing.scale(fvs, axis=1)
fvs = np.c_[fvs, ys]
"""
goodlength = max([len(row) for row in fvs])
goodis = [i for i in range(len(row)) if len(fvs[i]) == goodlength]
fvs2 = np.array([fvs[i] for i in goodis])
goodys = [ys[i] for i in goodis]

fvs3 = np.c_[fvs2, goodys]


fvsfile = open('fvs2.pkl','w')
pickle.dump([fvs, goodis, fvs3], fvsfille)
fvsfile.close()

fvsfile = open('fvs2.pkl','r')
fvs = pickle.load(fvsfile)[2]
fvsfile.close()
"""

tr_len = len(fvs)/2
val_len = len(fvs)/4

fvs = np.random.permutation(fvs)

#fvs = np.delete(fvs, [0,12], axis=1)

validation = fvs[tr_len:tr_len + val_len]

alphas = [10** i for i in range(-8,10)]
models = [lasso(fvs[:tr_len], fvs[tr_len:tr_len + val_len,], alpha) for alpha in alphas]



"""
for i in [0, 12]:
    model = lasso2(fvs[:tr_len], fvs[tr_len:tr_len + val_len], 0.0, i)
#    print model.score(fvs[tr_len:,i:i+1], fvs[tr_len:,-1])
#    print [model.score(validation[:,i:i+1], validation[:, -1]) for model in models]

scores = []
i=0
trials = 1000
for j in xrange(trials):
    fvs = np.random.permutation(fvs)
    model = lasso2(fvs[:tr_len], fvs[tr_len:tr_len + val_len], 0.0, i)
    scores.append(model.score(fvs[tr_len:,i:i+1], fvs[tr_len:,-1]))
print sum(scores)/trials
    
for i in range(7502):

    if max([model.score(validation[:,i:i+1], validation[:, -1]) for model in models]) > .3:
        print i
#
"""
### Volume plots:
"""
plt.scatter(fvs[:,12], fvs[:,-1])
plt.xlabel("RMS time homogeneity")
plt.ylabel("Texture realism scores")
plt.savefig("figures/rms_time_homogeneity.png")
plt.gcf().clear()

plt.scatter(fvs[:,0], fvs[:,-1])
plt.xlabel("RMS energy")
plt.ylabel("Texture realism scores")
plt.savefig("figures/rms_energy.png")
plt.gcf().clear()

plt.scatter(fvs[:,0], fvs[:,12])
plt.xlabel("RMS energy")
plt.ylabel("RMS time homogeneity")
plt.savefig("figures/rms_homogeneity_vs_energy.png")
plt.gcf().clear()
"""
