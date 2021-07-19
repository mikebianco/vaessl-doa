"""
Code for evaluating fully supervised convolution neural network (CNN) model for
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
import json
import torch
import numpy as np
import time
from utils.networks import CNN_yx, input_test
import utils.data_cls as data_cls
from cnn_train import classifier_fn, get_accuracy


def get_mae(data_loader, model, classes):
    """
    compute the doa mae over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    model.eval()   # call to evaulate mode with dropout

    error=0
    nData=0

    for (xs,ys) in data_loader:
        pred = classifier_fn(model,xs)
        true = ys
        estDOAs = classes[pred]
        trueDOAs= classes[true]
        error += torch.sum(torch.abs(estDOAs-trueDOAs))
        nData += len(trueDOAs)

    return error/nData


def get_mae_off(data_loaders, model, classes, classes_off):  # off-grid mae
    """
    compute the doa mae over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    model.eval()   # call to evaulate mode with dropout

    error=0
    nData=0

    for _,loader in data_loaders.items():

        for (xs,ys) in loader:
            pred = classifier_fn(model,xs)
            true = ys
            estDOAs = classes[pred]
            trueDOAs= classes_off[true]
            error += torch.sum(torch.abs(estDOAs-trueDOAs))
            nData += len(trueDOAs)

    return error/nData




def eval(args, model_dict, data_loader, labels_loader):
    """
    evaulate trained CNN model for DOA estimation
    """


    # instantiating model
    model = CNN_yx(x_size=(model_dict['n_seq_frames'],model_dict['num_bins']),
                        y_size=model_dict['output_size'],cnn_sup=True).to(device)

    model.load_state_dict(model_dict['model_state_dict'])

    if args.off_grid:
        mae = get_mae_off(data_loader, model, labels_loader['model'], labels_loader['off grid'])
        print('mae off grid = ',mae)

    else:
        mae = get_mae(data_loader['test'], model, labels_loader)
        accuracy = get_accuracy(data_loader['test'], model)
        print('mae = ',mae)
        print('accuracy = ',accuracy)


if __name__ == "__main__":
    with open('default_paths.json') as f:
        default_dict = json.load(f)


    parser = argparse.ArgumentParser(description="supervised CNN")

    parser.add_argument('-cid','--cuda-id', default = None, type = int,
                        help="use GPU(s) to speed up training")
    parser.add_argument('-mod','--model-file', default = default_dict['path_model']+default_dict['cnn_model'], help="trained model file")
    parser.add_argument('-ed','--eval-data', default = default_dict['path_data']+default_dict['data_valid'])
    parser.add_argument('-og','--off-grid',action = 'store_true')

    args = parser.parse_args()

    model_dict = torch.load(args.model_file)

    data_obj    = data_cls.DataClass(path=args.eval_data,addNoise=True,cuda_id=args.cuda_id,noiseSeed=0,
                             loader_shuffle = True, norm_factor = np.pi)

    data_loader = data_obj.get_cnn_data(nLabels=model_dict['sup_num'],nframes=model_dict['n_seq_frames'],
                                                    nBins=model_dict['num_bins'])

    if not args.off_grid:
        assert model_dict['output_size'] == len(data_obj._label_set), "mismatched output size"
        assert model_dict['label set'] == list(data_obj._label_set),  "mismatched label set"

    if args.cuda_id:
        device = 'cuda:'+ str(args.cuda_id)
    else:
        device = 'cpu'

    if args.off_grid:
        labels_loader = {}
        labels_loader['model']    = torch.tensor(np.array(model_dict['label set'])).to(device)
        labels_loader['off grid'] = torch.tensor(np.array(data_obj._label_set)).to(device)
    else:
        labels_loader = torch.tensor(np.array(model_dict['label set'])).to(device)

    eval(args, model_dict, data_loader, labels_loader)
