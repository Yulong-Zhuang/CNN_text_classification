import os

import numpy as np
import matplotlib.pyplot as plt

class Training_Preprocessor():
    def __init__(self, Path, ratio, output_folder):
        self.Path = Path
        self.output_folder = output_folder
        with open(self.Path+'xtrain_obfuscated.txt') as f:
            self.sent_raw = f.read().split('\n')
        with open(self.Path+'ytrain.txt') as f:
            self.label_raw = f.read().split('\n')
        self._raw_prop()
        self._sample_ana()
        self._train_test_split(ratio)

    def _raw_prop(self):
        '''Get matrix of sentence character sequence, frequency (features)'''
        print('preprocing raw data')
        sent = self.sent_raw[:-1]
        label = self.label_raw[:-1]
        self.s_lens = []
        for i in sent:
            self.s_lens.append(len(i))

        label_int = []
        for i in label:
            label_int.append(np.int(i))
        self.label_int = np.array(label_int)

        max_char = 26
        self.max_len = max(self.s_lens)
        X_sent = []
        for i in range(len(sent)):
            temp = np.zeros((max_char, self.max_len))
            for j in range(len(sent[i])):
                temp[ord(sent[i][j])-97][j] = 1
            X_sent.append(temp)
        self.X_sent_mat = np.array(X_sent)
        self.X_label_mat = np.zeros((len(self.label_int), max(self.label_int)+1))
        for i in range(len(sent)):
            self.X_label_mat[i, self.label_int[i]] = 1

        self.X_feature = self.X_sent_mat.sum(2)

    def _sample_ana(self):
        '''Analyzing sample'''
        print('visualizing training data')
        hist_per_sent = self.X_feature
        plt.figure(figsize=(10,8))
        for i in range(12):
            ng_i = np.where(self.label_int == i)[0]
            hist_per_sent_gpi = hist_per_sent[ng_i]
            plt.plot(hist_per_sent_gpi.mean(0) - hist_per_sent.mean(0), label='novel'+str(i))
        plt.legend()
        plt.xlabel('characters')
        plt.ylabel('feq_per_novel - mean_feq')
        plt.tight_layout()
        plt.savefig(self.output_folder+'char_freq_per_novel.png')

        plt.figure(figsize=(10,6))
        plt.subplot(121)
        plt.hist(self.s_lens, bins=100)
        plt.xlabel('characters per sentence')
        plt.ylabel('N')
        plt.tight_layout()
        plt.subplot(122)
        plt.hist(self.label_int, bins=12)
        plt.xlabel('Novel')
        plt.ylabel('N')
        plt.tight_layout()
        plt.savefig(self.output_folder+'Histograms.png')

    def _train_test_split(self, ratio=0.8):
        print('spliting training data')
        self.training_size = np.int(np.ceil(self.X_sent_mat.shape[0]*ratio))
        self.training_sent = self.X_sent_mat[:self.training_size]
        self.testing_sent = self.X_sent_mat[self.training_size:]
        self.training_feature = self.X_feature[:self.training_size]
        self.testing_feature = self.X_feature[self.training_size:]
        self.training_label = self.X_label_mat[:self.training_size]
        self.testing_label = self.X_label_mat[self.training_size:]
        self.training_label_int = self.label_int[:self.training_size]
        self.testing_label_int = self.label_int[self.training_size:]
        plt.figure(figsize=(8,6))
        plt.hist(self.training_label_int, bins=12, label='training_set')
        plt.hist(self.testing_label_int, bins=12, label='testing_set')
        plt.legend()
        plt.xlabel('Novel')
        plt.ylabel('N')
        plt.tight_layout()
        plt.savefig(self.output_folder+'training_testing_split_summary.png')



class Fitting_Preprocessor():
    def __init__(self, Path, model_max_len):
        with open(Path+'xtest_obfuscated.txt') as f:
            self.test_sent_raw = f.read().split('\n')
        self.max_len = model_max_len
        self.max_char = 26
        self._raw_prop()


    def _raw_prop(self):
        '''Get matrix of sentence character sequence, frequency (features)'''
        print('preprocing raw data')
        sent = self.test_sent_raw[:-1]

        T_sent = []
        for i in range(len(sent)):
            temp = np.zeros((self.max_char, self.max_len))
            for j in range(len(sent[i])):
                temp[ord(sent[i][j])-97][j] = 1
            T_sent.append(temp)
        self.T_sent_mat = np.array(T_sent)
        self.T_feature = self.T_sent_mat.sum(2)
