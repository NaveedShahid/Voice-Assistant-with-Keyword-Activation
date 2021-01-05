import pickle as cPickle
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from python_speech_features import delta
from python_speech_features import mfcc
import parameters as p
import warnings
import os
warnings.filterwarnings("ignore")

def train_model(speaker,source,dest):
    features = np.asarray(())
    files = [os.path.join(source,fname) for fname in os.listdir(source) if fname.endswith('.wav')]
    train_models = [os.path.join(dest,fname) for fname in os.listdir(dest) if fname==str(speaker+'.gmm')]
    if len(train_models) !=0:
        return True
    elif len(files) == 0:
        print("No WAV files found in the source directory.....")
        return False
    count = 0
    for f in sorted(files):
        if count<=len(os.listdir(source)):
            count = count + 1
            audio, sr = sf.read(f)
            vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
            vector = preprocessing.scale(vector)
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

    gmm = GaussianMixture(n_components = 16,covariance_type='diag',n_init = 3)
    gmm.fit(features)

    # dumping the trained gaussian model
    picklefile = speaker +".gmm"
    cPickle.dump(gmm,open(os.path.join(dest,picklefile),'wb'))
    return True
    
def recognize_command(speaker,test_file):
    modelpath = os.path.join(p.USER_FOLDER,speaker,"command_models/")

    gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    #Load the Gaussian Models
    models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    commands   = [fname.split("/")[-1].split(".gmm")[0] for fname
                  in gmm_files]

    audio, sr = sf.read(test_file)
    # extract 20 dimensional MFCC features
    vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
    log_likelihood = np.zeros(len(models))
    for i in range(len(models)):
        gmm    = models[i]         
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    score = log_likelihood
    winner = np.argmax(log_likelihood)

    if score > -5000:
        print("\tDetected command: ", commands)    
        print("\tScore: ", score)
    else:
        print("\tNo command detected, try again")    
        print("\tScore: ", score)
        
    return commands[winner]

def verify(speaker,command,test_file):
    modelpath = os.path.join(p.USER_FOLDER,speaker,"command_models/",command)
    if not os.path.exists(modelpath):
        print("Could not recognize a registered command. Please try again")
        return False

    audio, sr = sf.read(test_file)
    # extract 20 dimensional MFCC features
    vector = mfcc(audio,sr,0.025,0.01,20,appendEnergy=True)
 
    gmm = cPickle.load(open(modelpath,'rb'))       
    scores = np.array(gmm.score(vector))
    log_likelihood = scores.sum()
    score = log_likelihood
    winner = np.argmax(log_likelihood)

#     if score > -10000:
    print("\tDetected command: ", commands)    
    print("\tScore: ", score)
#     else:
#         print("\tNo command detected, try again")    
#         print("\tScore: ", score)
    return commands[winner]