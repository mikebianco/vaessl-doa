"""
Neural networks for semi-supervised variational autoencoder (VAE-SSL) model

The method implemented here is described in:
1. M.J. Bianco, S. Gannot, E. Fernandez-Grande, P. Gerstoft, "Semi-supervised
source localization in reverberant environments," IEEE Access, Vol. 9, 2021.
DOI: 10.1109/ACCESS.2021.3087697

If you find this code usefult for your research, please cite (1)
Michael J. Bianco, July 2021
mbianco@ucsd.edu
"""

import torch
import torch.nn as nn
import numpy as np

def input_test(x_size,n_layer=2):
    size = np.array(list(x_size))
    size_out = []
    for k in range(n_layer):  # math for kernel size 3 and stride 2
        size -= 1
        size = size/ 2
        size_out.append(size)

    size_better = size.copy().astype(int)

    for k in range(n_layer):
        size_better = size_better*2
        size_better += 1

    msg = 'Invalid input size! '+'A better size is: '+ str(size_better)

    assert all(size%1 == 0.0), msg
    for i,k in enumerate(size_out): size_out[i] = k.astype(int)
    return size_out


class CNN_yx(nn.Module):
    def __init__(self, x_size=(31,127), y_size = 37, use_cuda=False, cuda_id = 0, filts1 = 32, filts2 = 64, cnn_sup=False):
        super().__init__()

        self.size_l = input_test(x_size=x_size)  # raises assertion error if shape is invalid for current CNN architecture

        self.filts1 = filts1
        self.filts2 = filts2

        self.conv1 = nn.Conv2d(1, self.filts1, 3, stride =2 )
        self.conv2 = nn.Conv2d(self.filts1, self.filts2, 3, stride =2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self.filts2 * self.size_l[1][0] * self.size_l[1][1], 200)

        self.fc2 = nn.Linear(200,y_size)  # fc layer for connecting y to graph

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.x_size = x_size
        self.y_size = y_size

        self.batch_size_int = -2

        if cnn_sup: self.batch_size_int = 0

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda(cuda_id)

    def forward(self, x):

        batch_size = x.shape[self.batch_size_int]

        x = x.reshape(-1,1,self.x_size[0],self.x_size[1])

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, self.filts2 * self.size_l[1][0] * self.size_l[1][1] )
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        x = x.reshape(-1,batch_size,self.y_size)
        x = x.squeeze()
        return x

    def reset(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class CNN_zxy(nn.Module):
    def __init__(self, x_size = (31,127), y_size = 37, z_dim = 2, use_cuda=False, cuda_id=0, filts1 = 32, filts2 = 64):
        super().__init__()

        self.size_l = input_test(x_size=x_size)  # raises assertion error if shape is invalid for current CNN architecture

        self.filts1 = filts1
        self.filts2 = filts2

        self.conv1 = nn.Conv2d(1, self.filts1, 3, stride =2 )
        self.conv2 = nn.Conv2d(self.filts1, self.filts2, 3, stride =2)

        self.fc1 = nn.Linear(self.filts2 * self.size_l[1][0] * self.size_l[1][1], 200)

        self.fc2 = nn.Linear(200,y_size)  # fc layer for connecting y to graph

        self.dropout = nn.Dropout(0.5)


        self.fc_y = nn.Linear(y_size,200)  # fc layer for connecting y to graph

        self.fc_loc = nn.Linear(200, z_dim)
        self.fc_scale = nn.Linear(200, z_dim)

        self.relu = nn.ReLU()
        self.x_size = x_size
        self.y_size = y_size
        self.z_dim = z_dim

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda(cuda_id)

    def forward(self, input):
        x = input[0]
        y = input[1]

        batch_size = x.shape[-2]

        x = x.reshape(-1,1,self.x_size[0],self.x_size[1])

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(-1, self.filts2 * self.size_l[1][0] * self.size_l[1][1] )

        x = self.dropout(x)
        x = self.fc1(x)
        y = self.fc_y(y)

        yx = self.relu(x+y)  # linking CNN and MLP for y
        z_loc = self.fc_loc(yx) # no activation on output
        z_exp = torch.exp(self.fc_scale(yx))

        z_loc = z_loc.view(-1,batch_size,self.z_dim)
        z_exp = z_exp.reshape(z_loc.shape)

        return z_loc, z_exp


    def reset(self):
        # explicity resetting parameters of the respective layers

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc_y.reset_parameters()
        self.fc_loc.reset_parameters()
        self.fc_scale.reset_parameters()

class CNN_xyz(nn.Module):  # x|y,z
    def __init__(self, x_size = (31,127), y_size = 37, z_dim = 2, use_cuda=False, cuda_id=0, filts1 = 32, filts2 = 64):
        super().__init__()

        self.size_l = input_test(x_size=x_size)  # raises assertion error if shape is invalid for current CNN architecture

        self.x_size = x_size


        self.filts1 = filts1
        self.filts2 = filts2

        self.fc_y = nn.Linear(y_size,200)
        self.fc_z = nn.Linear(z_dim,200)

        self.fc_yz = nn.Linear(200,self.filts2 * self.size_l[1][0] * self.size_l[1][1])

        self.conv1_T = nn.ConvTranspose2d(self.filts2, self.filts1, 3, stride = 2)
        self.conv2_T = nn.ConvTranspose2d(self.filts1, 1, 3, stride = 2)

        self.conv1_T_scale = nn.ConvTranspose2d(self.filts2, self.filts1, 3, stride = 2)
        self.conv2_T_scale = nn.ConvTranspose2d(self.filts1, 1, 3, stride = 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

        self.dropout = nn.Dropout(0.5)

        self.cuda_id = cuda_id



        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda(cuda_id)

    def forward(self, input):
        z_loc=input[0]
        y = input[1]
        batch_size = y.shape[-2]

        yz  = self.relu(self.fc_z(z_loc)+self.fc_y(y))
        yz  = self.relu(self.dropout(self.fc_yz(yz)))

        x  = yz.view(-1,self.filts2,self.size_l[1][0],self.size_l[1][1])

        x_loc  = self.relu(self.conv1_T(x))
        x_loc  = self.conv2_T(x_loc)

        x_log_scale = self.relu(self.conv1_T_scale(x))
        x_log_scale = self.conv2_T_scale(x_log_scale)

        lenx = self.x_size[0]*self.x_size[1]

        x_loc = x_loc.view(-1,batch_size,lenx) # no activation

        x_log_scale = x_log_scale.view(-1,batch_size,lenx)
        x_scale = torch.sigmoid(x_log_scale)*10.0
        x_loc = self.Tanh(x_loc)

        return x_loc,x_scale

    def reset(self):

        # explicity resetting parameters of the respective layers
        self.fc_y.reset_parameters()
        self.fc_z .reset_parameters()
        self.fc_yz.reset_parameters()
        self.conv1_T.reset_parameters()
        self.conv2_T.reset_parameters()
        self.conv1_T_scale.reset_parameters()
        self.conv2_T_scale.reset_parameters()
