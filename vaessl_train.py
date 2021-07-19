"""
Code for training semi-supervised variational autoencoder (VAE-SSL) model for
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
import json
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from utils.networks import input_test
import utils.data_cls as data_cls
from vaessl_core import SSVAE, run_inference_for_epoch, get_accuracy


def train(args,train_obj,valid_obj):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """

    data_loaders1 = train_obj.get_vaessl_data(nLabels=args.sup_num,batch_size=args.batch_size,nframes=args.n_seq_frames,nBins=args.num_bins)
    data_loaders2 = valid_obj.get_vaessl_data(nLabels=args.sup_num,batch_size=args.batch_size,nframes=args.n_seq_frames,nBins=args.num_bins)

    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    ss_vae = SSVAE(z_dim=args.z_dim,
                   cuda_id=args.cuda_id,
                   config_enum=args.enum_discrete,
                   aux_loss_multiplier=args.aux_loss_multiplier,
                   output_size = args.output_size,
                   input_size  = (args.n_seq_frames,args.num_bins))

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)

    PATH_temp = args.path_save+args.temp_name

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    guide = config_enumerate(ss_vae.guide, args.enum_discrete, expand=True)
    elbo = (TraceEnum_ELBO)(max_plate_nesting=1)
    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if args.aux_loss:
        elbo = Trace_ELBO()
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)


    early_stop_counter = 0
    acu_min = 0
    accu_save1 = []
    accu_save2 = []

    best_valid_acc, corresponding_test_acc = 0.0, 0.0

    sup_num_load   = len(data_loaders1["sup"])
    unsup_num_load = len(data_loaders1["unsup"])
    sup_num_data   = len(data_loaders1["sup"].dataset)
    unsup_num_data = len(data_loaders1["unsup"].dataset)

    periodic_interval_batches = (sup_num_load+unsup_num_load) // sup_num_load

    arg_dict = vars(args)
    arg_dict['label set']=train_obj._label_set   # saving the label set to model dictionary

    # run inference for a certain number of epochs
    for i in range(0, args.num_epochs):

        # get the losses for an epoch
        epoch_losses_sup, epoch_losses_unsup =\
        run_inference_for_epoch(data_loaders1, losses, periodic_interval_batches)  # using dataloader 1


        # compute average epoch losses i.e. losses per example
        # mjb: here v is an argument, left of colon is expression, which looks like it returns two items
        avg_epoch_losses_sup   = map(lambda v: v / sup_num_data, epoch_losses_sup) # map returns an iterator after applying given function to each item of given input
        avg_epoch_losses_unsup = map(lambda v: v / unsup_num_data, epoch_losses_unsup)

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup   = " ".join(map(str, avg_epoch_losses_sup))
        str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

        str_print = "{} epoch: avg losses- {}".format(i, "sup:{} unsup:{}".format(str_loss_sup, str_loss_unsup))

        validation_acc1 = get_accuracy(data_loaders1["sup"], ss_vae.classifier, ss_vae.num_classes)
        validation_acc2 = get_accuracy(data_loaders2["sup"], ss_vae.classifier, ss_vae.num_classes)  # now using supervised generalize set

        accu_save1.append(validation_acc1)
        accu_save2.append(validation_acc2)

        # saving model if accuracy increases
        loss_sup=np.array(str_loss_sup.split(' ')).astype('float')
        acu_check = validation_acc2
        if acu_check > acu_min:
            model_state = ss_vae.state_dict()
            torch.save(model_state, PATH_temp)
            acu_min = acu_check
            print('SAVING MODEL TO TEMP')
            early_stop_counter = 0

        early_stop_counter += 1

        # this test accuracy is only for logging, this is not used
        # to make any decisions during training
        test_accuracy = validation_acc1
        str_print += " valid accuracy1 {}".format(validation_acc1)
        str_print += " valid accuracy2 {}".format(validation_acc2)

        print(str_print)

        if validation_acc2==1. or early_stop_counter >99:
            break


    accu_save1 = np.array(accu_save1)
    accu_save2 = np.array(accu_save2)

    # saving final model
    model_state = torch.load(PATH_temp)

    arg_dict['model_state_dict'] = model_state
    arg_dict['train acc epoch'] = accu_save1  # training history
    arg_dict['valid acc epoch'] = accu_save2

    timeLocal=time.localtime()
    timePrint=time.asctime(timeLocal).replace(" ", "_").replace(":", "")


    PATH_model = 'vaessl_model_{}-{}-{}_J{}_{}.cpkt'.format(args.learning_rate, args.num_epochs,\
                                                                   args.batch_size, args.sup_num, timePrint)
    torch.save(arg_dict,args.path_save+PATH_model)


if __name__ == "__main__":
    # importing default paths
    with open('default_paths.json') as f:
        default_dict = json.load(f)

    parser = argparse.ArgumentParser(description="VAE-SSL")

    parser.add_argument('-cid','--cuda-id', default = None, type = int,
                        help="use GPU(s) to speed up training")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int,
                        help="number of epochs to run")
    parser.add_argument('-al', '--aux-loss', default = True,
                        help="whether to use the auxiliary loss from NIPS 14 paper "
                             "(Kingma et al). It is used by default ")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=5000., type=float,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', default="parallel",
                        help="parallel, sequential or none. uses parallel enumeration by default")
    parser.add_argument('-nsup', '--sup-num', default=1000,
                        type=int, help="supervised amount of the data i.e. "
                                         "how many of the RTF-phase sequences have supervised labels")
    parser.add_argument('-zd', '--z-dim', default=50, type=int,
                        help="size of the tensor representing the latent variable z ")
    parser.add_argument('-lr', '--learning-rate', default=0.00005, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=256, type=int,
                        help="number of RTFs (and labels) to be considered in a batch")
    parser.add_argument('--seed', default=None, type=int,
                        help="seed for controlling randomness in this example")
    parser.add_argument('--output-size', default=19, type=int,
                        help="number of DOAs")
    parser.add_argument('-nbin', '--num-bins',default = 127, type=int)
    parser.add_argument('-nseq','--n-seq-frames', default = 31, type=int)
    parser.add_argument('-dt','--train-data', default = default_dict['path_data']+default_dict['data_train'])
    parser.add_argument('-dv','--valid-data', default = default_dict['path_data']+default_dict['data_valid'])
    parser.add_argument('-ps','--path-save', default= default_dict['path_model'], help="path for saving model")
    parser.add_argument('-tn','--temp-name', default = 'temp.cpkt', help='temporary model file name')

    args = parser.parse_args()

    input_test((args.n_seq_frames,args.num_bins))  # testing if the input dimensions will work with the networks

    # getting data and loaders
    data_obj_train = data_cls.DataClass(path=args.train_data,addNoise=True,cuda_id=args.cuda_id,noiseSeed=0,
                             loader_shuffle = True, norm_factor = np.pi)
    data_obj_valid = data_cls.DataClass(path=args.valid_data,addNoise=True, cuda_id=args.cuda_id,noiseSeed=1,
                             loader_shuffle = True, norm_factor = np.pi)

    train(args, data_obj_train, data_obj_valid)
