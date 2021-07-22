"""
Code for training fully-supervised convolutional neural network (CNN) model for
source localization in reverberant environments

The method implemented here is described in:
1. M.J. Bianco, S. Gannot, E. Fernandez-Grande, P. Gerstoft, "Semi-supervised
source localization in reverberant environments," IEEE Access, Vol. 9, 2021.
DOI: 10.1109/ACCESS.2021.3087697

The code is based on the Pyro probabilistic programming library and Pytorch.
2. E. Bingham et al., "Pyro: Deep Universal Probabilistic Programming,"
Journal of Machine Learning Research, 2018.
3. A. Paszke et al., "Pytorch: An imperative style, high-performance deep
learning library," Proc. Adv. Neural Inf. Process. Syst., 2019, pp. 8024–8035.

If you find this code usefult for your research, please cite (1)--(3).
Michael J. Bianco, July 2021
mbianco@ucsd.edu
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time

from utils.networks import CNN_yx, input_test
import utils.data_cls as data_cls
import torch.optim as optim
import sys
import json


def train(args,train_obj,valid_obj):
    """
    train fully-supervised CNN on labeled RTF-phase sequences
    """

    data_loaders1 = train_obj.get_cnn_data(nLabels=args.sup_num,batch_size=args.batch_size,nframes=args.n_seq_frames,nBins=args.num_bins)
    data_loaders2 = valid_obj.get_cnn_data(nLabels=args.sup_num,batch_size=args.batch_size,nframes=args.n_seq_frames,nBins=args.num_bins)

    loader_train = data_loaders1['sup'] # using only labeled frames for CNN
    loader_valid = data_loaders2['sup']

    if args.cuda_id:
        device = 'cuda:'+ str(args.cuda_id)
    else:
        device = 'cpu'

    use_gpu = True
    cnn = CNN_yx(x_size=(args.n_seq_frames,args.num_bins),y_size=len(train_obj._label_set),cnn_sup=True).to(device)

    cnn.reset()

    LR = args.learning_rate
    NUM_EPOCHS = args.num_epochs

    criterion = nn.CrossEntropyLoss()
    optimizer=optim.Adam(cnn.parameters(),lr=LR)

    running_loss_save=[]
    running_loss_test_save=[]
    acc_save = []
    PATH_temp = 'save_models/cnn_temp.cpkt'
    loss_min=float('inf')
    acc_max=0.
    patience_count = 0

    cnn.train()

    arg_dict = vars(args)
    arg_dict['label set']=train_obj._label_set


    for epoch in range(NUM_EPOCHS):

        running_loss = 0.0
        for i, data in enumerate(loader_train, 0):

            inputs = data[0]
            labels = data[1]

            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss_save.append((running_loss/len(loader_train)))

        predictions, actuals = [], []
        numCases = 0

        acc_valid= get_accuracy(loader_valid, cnn)

        if acc_valid > acc_max:
            arg_dict['model_state_dict'] = cnn.state_dict()
            torch.save(arg_dict, PATH_temp)
            acc_max = acc_valid
            print('saving model to temp')
            patience_count = 0

        patience_count += 1

        if patience_count > 99:
            break



        str_print = '{} train loss: {}'.format(epoch + 1, running_loss / len(loader_train))
        str_print += ' valid accu: {}'.format(acc_valid)
        print(str_print)


    print('Finished Training')

    model_state = arg_dict['model_state_dict']
    for key,value in model_state.items():
        model_state[key] = value.cpu()  # pushing to cpu device as default

    arg_dict['model_state_dict']=model_state


    # saving final model
    arg_dict = torch.load(PATH_temp)

    timeLocal=time.localtime()
    timePrint=time.asctime(timeLocal).replace(" ", "_").replace(":", "")


    PATH_model = 'cnn_model_{}-{}-{}_J{}_{}.cpkt'.format(args.learning_rate, args.num_epochs,\
                                                                   args.batch_size, args.sup_num, timePrint)
    torch.save(arg_dict,args.path_save+PATH_model)


def classifier_fn(model, xs):
    """
    give one-hot DOAs from RTF-phase input

    :param xs: a batch of scaled vectors of RTF-phase sequences
    :return: a batch of the corresponding DOA labels (one-hot representation)
    """
    alpha = model.forward(xs)
    res, ind = torch.topk(alpha, 1)
    ys = ind.flatten()
    return ys


def get_accuracy(data_loader, model):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []
    numCases = 0

    model.eval()

    # use the appropriate data loader
    for (xs, ys) in data_loader:
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(model,xs))
        actuals.append(ys)
        numCases += len(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = pred[i] == act[i]
            accurate_preds += v


    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / numCases # mjb (len(predictions) * batch_size)
    return accuracy


if __name__ == "__main__":
    # importing default paths
    with open('default_paths.json') as f:
        default_dict = json.load(f)

    parser = argparse.ArgumentParser(description="Supervised CNN")

    parser.add_argument('-cid','--cuda-id', default = None, type = int,
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int,
                        help="number of epochs to run")
    parser.add_argument('-nsup', '--sup-num', default=1000,
                        type=int, help="supervised amount of the data i.e. "
                                         "how many of the RTF-phase sequences have supervised labels")
    parser.add_argument('-lr', '--learning-rate', default=0.00005, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=256, type=int,
                        help="number of RTF-phase sequences (and DOAs) to be considered in a batch")
    parser.add_argument('--output-size', default=19, type=int,
                        help="number of DOAs")
    parser.add_argument('-nbin', '--num-bins',default = 127, type=int)
    parser.add_argument('-nseq','--n-seq-frames', default = 31, type=int)
    parser.add_argument('-dt','--train-data', default = default_dict['path_data']+default_dict['data_train'])
    parser.add_argument('-dv','--valid-data', default = default_dict['path_data']+default_dict['data_valid'])
    parser.add_argument('-ps','--path-save',default=default_dict['path_model'], help="path for saving model")

    args = parser.parse_args()

    input_test((args.n_seq_frames,args.num_bins))  # testing if the input dimensions will work with the networks


    # getting data and loaders

    data_obj_train = data_cls.DataClass(path=args.train_data,addNoise=True,cuda_id=args.cuda_id,noiseSeed=0,
                             loader_shuffle = True, norm_factor = np.pi)
    data_obj_valid = data_cls.DataClass(path=args.valid_data,addNoise=True, cuda_id=args.cuda_id,noiseSeed=1,
                             loader_shuffle = True, norm_factor = np.pi)

    train(args, data_obj_train, data_obj_valid)
