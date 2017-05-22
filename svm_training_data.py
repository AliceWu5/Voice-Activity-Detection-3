import numpy as np
import sys
from sklearn import svm


def readInputData():
	voice_inp = np.loadtxt('voice.txt')
	nonvoice_inp = np.loadtxt('nonvoice.txt')
	return voice_inp, nonvoice_inp

def main():
	vc_inp, nonvc_inp = readInputData()
	vc_lab = np.ones((len(vc_inp), 1))
	nonvc_lab = np.zeros((len(nonvc_inp), 1))
	np.random.shuffle(vc_inp)
	np.random.shuffle(nonvc_inp)
	
	lenvc, lennonvc = len(vc_inp), len(nonvc_inp)
	training_lenvc, training_lennonvc = lenvc*9/10, lennonvc*9/10
	test_lenvc, test_lennonvc = lenvc-training_lenvc, lennonvc-training_lennonvc

	training_set = np.vstack((vc_inp[:training_lenvc], nonvc_inp[:training_lennonvc]))
	training_label = np.vstack((vc_lab[:training_lenvc], nonvc_lab[:training_lennonvc]))
	
	clf = svm.SVC()
	clf.fit(training_set, training_label)
	
	positive_predict = clf.predict(vc_inp[training_lenvc:])
	negative_predict = clf.predict(nonvc_inp[training_lennonvc:])
	
	true_pos = np.count_nonzero(positive_predict)
	false_neg = test_lenvc-true_pos
	false_pos = np.count_nonzero(negative_predict)
	true_neg = test_lennonvc-false_pos
	pre = float(true_pos)/(true_pos+false_pos)	# precision
	rec = float(true_pos)/(true_pos+false_neg)	# recall
	f = open('report.txt', 'w')
	f.write('Precision: '+str(true_pos)+'/'+str(true_pos+false_pos)+'='+str(pre)+'\n')
	f.write('Recall: '+str(true_pos)+'/'+str(true_pos+false_neg)+'='+str(rec)+'\n')
	f.close()
	

if (__name__ == '__main__'):
	main()
