import os
import argparse

import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional

from Preprocessing import Training_Preprocessor, Fitting_Preprocessor
from CNN_model import Novel_det

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


#Path = '/Users/yulongzhuang/Documents/Applications/SAP/Challenge/offline_challenge/'

class Novel_classifier():
    def __init__(self, Path, output_folder):
        self.Path = Path
        self.output_folder = output_folder
        self.TP = Training_Preprocessor(self.Path, 0.8, self.output_folder)
        self.TT = Fitting_Preprocessor(self.Path, self.TP.max_len)
        self.model = Novel_det(self.TP.max_len).to(device)

    def loss_function(self, pred_x, x):
            #BCE = functional.binary_cross_entropy(pred_x, x, reduction='sum')
            CCE = (-(pred_x + 0.000001).log() * x).sum(dim=1).mean()
            #NLL = nn.NLLLoss()(torch.log(pred_x), x.argmax(1))
            BSE = torch.sum((pred_x - x)**2)/pred_x.size()[0]
            return BSE + CCE#+ CCE #BCE + BSE, BCE,

    def evaluation(self, pred, label):
        pred_novel = pred.argmax(1)
        label_novel = label.argmax(1)
        correctness = []
        for i in range(len(pred_novel)):
            correctness.append(pred_novel[i] == label_novel[i])

        accuracy = np.sum(correctness)/len(correctness)
        return accuracy

    def _test(self):
        self.model.eval()
        pred_all = []
        loss_all = []
        testing_size = len(self.TP.testing_label)
        for i in range(testing_size // self.batch_size):
            # Local batches and labels
            sent_i = self.TP.testing_sent[i*self.batch_size:(i+1)*self.batch_size]
            sentence = torch.from_numpy(sent_i)
            sentence = sentence.float().to(device)
            label_i = self.TP.testing_label[i*self.batch_size:(i+1)*self.batch_size]
            label = torch.from_numpy(label_i)
            label = label.float().to(device)
            pred = self.model(sentence)
            loss = self.loss_function(pred, label)

            pred_all = pred_all + list(pred.detach().cpu().clone().numpy())
            loss_all.append(loss.detach().cpu().clone().numpy())

        test_accuracy = (self.evaluation(np.array(pred_all), self.TP.testing_label))
        test_loss_ave = (np.mean(loss_all))
        return test_accuracy, test_loss_ave, pred_all

    def training(self, n_epochs=60, batch_size=64, n_freeze=100):

        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        print('Start training')

        traning_size = len(self.TP.training_label)
        testing_size = len(self.TP.testing_label)
        #print(n_epochs,' intens_size =', intensize)
        self.Training_accuracy = []
        self.Training_loss_ave = []
        self.test_accuracy = []
        self.test_loss_ave = []
        for epoch in range(n_epochs):
            pred_all = []
            loss_all = []
            arr = np.arange(traning_size)
            training_label_r = self.TP.training_label[arr]
            training_sent_r = self.TP.training_sent[arr]
            for i in range(traning_size // self.batch_size):
                # Local batches and labels
                sent_batch = training_sent_r[i*self.batch_size:(i+1)*self.batch_size]
                sentence = torch.from_numpy(sent_batch)
                sentence = sentence.float().to(device)

                label_batch = training_label_r[i*self.batch_size:(i+1)*self.batch_size]
                label = torch.from_numpy(label_batch)
                label = label.float().to(device)

                pred = self.model(sentence)
                loss = self.loss_function(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_all = pred_all + list(pred.detach().cpu().clone().numpy())
                loss_all.append(loss.detach().cpu().clone().numpy())

                if i%100 == 0:
                    to_print = "Epoch[{}/{}] Loss: {:.3f}".format(epoch+1,
                                            n_epochs, loss.data.item())
                    print(to_print)

            self.Training_accuracy.append(self.evaluation(np.array(pred_all), self.TP.training_label))
            self.Training_loss_ave.append(np.mean(loss_all))
            test_accuracy_i, test_loss_ave_i, test_pred_all = self._test()
            self.test_accuracy.append(test_accuracy_i)
            self.test_loss_ave.append(test_loss_ave_i)

            if (epoch > n_freeze):
                if (Training_loss_ave[epoch] < 0.03):
                    print('freezing Conv1d_layers')
                    self.model.cnn_1.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_1.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_2.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_2.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_3.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_3.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_4.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_4.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_5.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_5.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_6.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_6.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_7.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_7.cnnnet[0].bias.requires_grad = False
                    self.model.cnn_8.cnnnet[0].weight.requires_grad = False
                    self.model.cnn_8.cnnnet[0].bias.requires_grad = False



            print("Epoch[{}/{}] Accuracy: {:.3f} Loss: {:.3f}".format(epoch+1,n_epochs,self.Training_accuracy[epoch],self.Training_loss_ave[epoch]))
            print("Test Accuracy: {:.3f} Loss: {:.3f}".format(self.test_accuracy[epoch],self.test_loss_ave[epoch]))

        torch.save(self.model.state_dict(), self.output_folder + 'trained_model_dict')
        print('Done training!')
        plt.figure(figsize=(10,6))
        plt.subplot(121)
        plt.plot(self.Training_accuracy, label='training')
        plt.plot(self.test_accuracy, label='testing')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.subplot(122)
        plt.plot(self.Training_loss_ave, label='training')
        plt.plot(self.test_loss_ave, label='testing')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.savefig(self.output_folder + 'training_summary.png')

    def fitting(self, batch_size=100):
        self.model.load_state_dict(torch.load(self.output_folder + 'trained_model_dict'))
        self.model.eval()
        pred_all = []
        testing_size = len(self.TT.T_sent_mat)
        for i in range(testing_size // batch_size):
            # Local batches and labels
            sent_i = self.TT.T_sent_mat[i*batch_size:(i+1)*batch_size]
            sentence = torch.from_numpy(sent_i)
            sentence = sentence.float().to(device)
            pred = self.model(sentence)

            pred_all = pred_all + list(pred.detach().cpu().clone().numpy())

        self.pred_all = np.array(pred_all)
        self.pred_novel = self.pred_all.argmax(1)
        return self.pred_novel, self.pred_all

def main():
    parser = argparse.ArgumentParser(description='path to the data folder')
    parser.add_argument('-p', '--path', default='/home/zhuangyu/GAN_data/DATA/', help='path to the data files')
    parser.add_argument('-o', '--output_name', default='002', help='path to the data files')
    args = parser.parse_args()
    output_name = args.output_name
    Path = args.path
    output_folder = Path + '/model_' + output_name + '/'
    print('Reading data from: ' + Path)
    try:
        os.mkdir(output_folder)
        print('create dir: ' + output_folder)
    except:
        print('cannot create dir')
    Cfer = Novel_classifier(Path, output_folder)
    Cfer.training()
    pred_novel, pred_mat = Cfer.fitting() #model_001_dict
    with open(output_folder + 'ytest.txt', 'w') as f:
        for item in pred_novel:
            f.write("%s\n" % item)

if __name__ == '__main__':
    main()
