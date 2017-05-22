import numpy as np
import scipy.io.wavfile as wav
import math
import sys
import librosa
import sklearn
from os import listdir

def getMFCCVector(wav_file):
	(rate, signal) = wav.read(wav_file)
	mfcc_vec = librosa.feature.mfcc(y = signal, sr = rate)
	return np.transpose(sklearn.preprocessing.scale(mfcc_vec, axis=1))

def getData(wavdir):
	filenames = listdir(wavdir)
	input_vector = getMFCCVector(wavdir+filenames[0])
	for f in filenames[1:]:
		mfcc_vector = getMFCCVector(wavdir+f)
		input_vector = np.vstack((input_vector, mfcc_vector))
	return input_vector

def getAndPrintData():
	voice_inp = getData('Data/Voice/')
	np.savetxt('voice.txt', voice_inp[:30000])
	nonvoice_inp = getData('Data/NonVoice/')
	np.savetxt('nonvoice.txt', nonvoice_inp[:20000])

def main():
	getAndPrintData()

if (__name__ == "__main__"):
	main()	
