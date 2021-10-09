import torch
import torch.nn as nn
# Device configuration above 80%
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Cnn_net(nn.Module):
    """Conv1d_layers"""
    def __init__(self, kernel_size, max_len):
        super().__init__()
        self.cnnnet = nn.Sequential(
            nn.Conv1d(26, 256, kernel_size=kernel_size, stride=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.5), #0.5
            nn.MaxPool1d(max_len-kernel_size+1),
            nn.Flatten()
            )
    def forward(self, x):
        return self.cnnnet(x)

class Novel_det(nn.Module):
    def __init__(self, max_len):
        super(Novel_det, self).__init__()
        self.max_len = max_len

        self.cnn_1 = Cnn_net(3, self.max_len)
        self.cnn_2 = Cnn_net(4, self.max_len)
        self.cnn_3 = Cnn_net(5, self.max_len)
        self.cnn_4 = Cnn_net(6, self.max_len)
        self.cnn_5 = Cnn_net(7, self.max_len)
        self.cnn_6 = Cnn_net(8, self.max_len)
        self.cnn_7 = Cnn_net(9, self.max_len)
        self.cnn_8 = Cnn_net(10, self.max_len)
        self._fc()


    def _fc(self):
        self.fc = nn.Sequential(
            nn.Linear(1024*2, 1024), nn.ReLU(), nn.Dropout(p=0.3), #0.3
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=0.1), #0.1
            nn.Linear(512, 12),nn.Softmax()
            )

    def forward(self, x):
        x_char = x
        x01 = self.cnn_1(x_char)
        x02 = self.cnn_2(x_char)
        x03 = self.cnn_3(x_char)
        x04 = self.cnn_4(x_char)
        x05 = self.cnn_5(x_char)
        x06 = self.cnn_6(x_char)
        x07 = self.cnn_7(x_char)
        x08 = self.cnn_8(x_char)
        x1 = torch.cat((x01,x02,x03,x04,x05,x06,x07,x08), dim=1)
        x2 = self.fc(x1)
        return x2
