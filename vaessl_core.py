"""
Code implementing inference and generative models for  semi-supervised
variational autoencoder (VAE-SSL) approach to source localization in reverberant
environments

The method implemented here is described in:
1. M.J. Bianco, S. Gannot, E. Fernandez-Grande, P. Gerstoft, "Semi-supervised
source localization in reverberant environments," IEEE Access, Vol. 9, 2021.
DOI: 10.1109/ACCESS.2021.3087697

The code is based on the Pyro probabilistic programming library and Pytorch.
Copyright (c) 2017-2019 Uber Technologies, Inc.
SPDX-License-Identifier: Apache-2.0

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
import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.optim import Adam

from utils.networks import CNN_yx,CNN_zxy,CNN_xyz
import utils.data_cls as data_cls

class TruncatedNormal(dist.Rejector):
    def __init__(self, loc, scale,cuda_id):
        lim = 1.0
        min_x = torch.tensor(np.array([-lim]).astype('float32'))
        max_x = torch.tensor(np.array([lim]).astype('float32'))
        if cuda_id is not None:
            min_x = min_x.cuda(cuda_id)
            max_x = max_x.cuda(cuda_id)


        propose = dist.Normal(loc, scale)

        def log_prob_accept(x): # gives log acceptance (0 if outside, and 1 if inside truncated Gaussian)
            thresh_min = x >= min_x
            thresh_max = x <= max_x
            return (thresh_min*thresh_max).type_as(x).log()  # logical multiplcation, same as 'and'

        cdf_min = dist.Normal(loc, scale).cdf(min_x)
        cdf_max = dist.Normal(loc, scale).cdf(max_x)
        log_scale = torch.log(cdf_max-cdf_min)  # log total probability of acceptance
        super(TruncatedNormal, self).__init__(propose, log_prob_accept, log_scale)




class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder to generate RTF-phase sequences and localize acoustic sources

    :param output_size: size of the tensor representing the DOA
    :param input_size: size of the tensor representing the RTF-phase sequence
    :param z_dim: size of the tensor representing the latent random variable z
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """
    def __init__(self, output_size=10, input_size=(31,127), z_dim=50,
                 config_enum=None, aux_loss_multiplier=None, batch_size=256, cuda_id=None):

        super().__init__()

        use_cuda = False
        if cuda_id is not None: use_cuda = True
        # print(use_cuda, cuda_id)

        self.output_size = output_size
        self.input_size  = input_size
        self.z_dim = z_dim
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.batch_size = batch_size
        self.cuda_id = cuda_id
        self.num_classes = output_size

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim

        self.encoder_y = CNN_yx(use_cuda=self.use_cuda,y_size=self.output_size,cuda_id=self.cuda_id, x_size=self.input_size)
        self.encoder_z = CNN_zxy(use_cuda=self.use_cuda,y_size=self.output_size,z_dim=self.z_dim,cuda_id=self.cuda_id, x_size=self.input_size)
        self.decoder   = CNN_xyz(use_cuda=self.use_cuda,y_size=self.output_size,z_dim=self.z_dim,cuda_id=self.cuda_id, x_size=self.input_size)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda(self.cuda_id)

    def model(self, xs, ys=None, **kwargs):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # latent variable, factors generating RTF unrelated to source DOA
        p(y|x) = categorical(I/10.)     # which DOA (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # RTF-phase sequence
        loc is given by a neural network  `decoder`

        :param xs: a batch of RTF-phase sequences
        :param ys: (optional) a batch of the DOA labels, corresponding to input RTF-phase sequence
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("data"):

            # sample the latent variable from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (1.0 * self.output_size)

            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            x_loc, x_scale = self.decoder.forward([zs, ys])

            pyro.sample("x", TruncatedNormal(x_loc,x_scale,self.cuda_id).to_event(1), obs=xs)


    def guide(self, xs, ys=None, **kwargs):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer DOA from an RTF-phase sequence
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # obtain latent value corresponding to an RTF-phase sequence and DOA
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of RTF-phase sequences
        :param ys: (optional) a batch of the DOA labels, corresponding to input RTF-phase sequence
        :return: None
        """
        self.encoder_y.train()  # ensuring that network is trained (vs. eval)

        with pyro.plate("data"):

            # if the class label is not supervised, sample
            # (and score) the RTF-phase sequence with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))


            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


    def classifier(self, xs, eval = False):
        """
        estimate DOA from RTF-phase sequence

        :param xs: a batch of RTF-phase sequences
        :return: a batch of the corresponding DOAs (as one-hot encoding)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for RTF-sequence(s)
        if eval == True:
            self.encoder_y.eval()
        else:
            self.encoder_y.train()

        alpha = self.encoder_y.forward(xs)


        # get the index (DOA) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the indices to one-hot tensor(s)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None, **kwargs):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        self.encoder_y.train()

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None, **kwargs):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def reconstruct_img2(self, xs, ys):
        # encode RTF-phase sequence x
        z_loc, z_scale = self.encoder_z([xs, ys])
        # sample in latent space
        zs = dist.Normal(z_loc, z_scale).sample()

        # decode the RTF-phase sequence
        x_loc,x_scale = self.decoder([zs,ys])
        xr = TruncatedNormal(x_loc,x_scale,self.cuda_id).sample()

        return xr, zs, x_loc, x_scale

    def cond_sample(self, ys):
        # sample from p(z)
        ys = ys.unsqueeze(0) # adding singleton dimension to interface with CNNs
        prior_loc  = torch.zeros(ys.shape[-2], self.z_dim)
        prior_scale = torch.ones(ys.shape[-2], self.z_dim)

        # sample in latent space
        zs = dist.Normal(prior_loc,prior_scale).sample()
        zs = zs.cuda(self.cuda_id)

        # decode the RTF-phase sequence
        x_loc,x_scale = self.decoder([zs,ys])
        xr = TruncatedNormal(x_loc,x_scale,self.cuda_id).sample()

        return xr, zs, x_loc, x_scale



def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys) = next(unsup_iter)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    return epoch_losses_sup, epoch_losses_unsup


def get_loss_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1

        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].evaluate_loss(xs, ys)
                epoch_losses_sup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup


def get_accuracy(data_loader, classifier_fn, num_classes):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []
    numCases = 0

    # use the appropriate data loader
    for (xs, ys) in data_loader:
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(xs,eval=True))  # eval to suspend dropout sampling in trained network
        actuals.append(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):         # for each case from the batch
            v = torch.sum(pred[i] == act[i])  # matching classes
            accurate_preds += (v.item() == num_classes)
            numCases += 1

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / numCases
    return accuracy


def get_mae(data_loader, vae_model, classes):
    """
    compute the doa mae over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    error=0
    nData=0

    vae_model.eval()  # eval to suspend dropout sampling in trained network

    for i, data in enumerate(data_loader, 0):
        pred = vae_model(data[0])
        estInds = pred.argmax(axis=1)
        trueInds = data[1].argmax(axis=1)
        estDOAs = classes[estInds]
        predictions.extend(estDOAs.detach().cpu().flatten().tolist())
        trueDOAs= classes[trueInds]
        actuals.extend(trueDOAs.detach().cpu().flatten().tolist())
        error += torch.sum(torch.abs(estDOAs-trueDOAs))
        nData += len(trueDOAs)

    mae = error/nData

    return mae, [predictions,actuals]

def get_mae_off(data_loader, vae_model, classes_dict):
    """
    off-grid/ off-range processing compute the doa mae over the supervised training set or the testing set
    """

    predictions, actuals = [], []

    error=0
    nData=0

    classes     = classes_dict['model']
    classes_off = classes_dict['off grid']

    vae_model.eval()  # eval to suspend dropout sampling in trained network
    loader_keys = ["sup","test"]  # using both sup and test for calculating error

    for k in loader_keys:
        for i, data in enumerate(data_loader[k], 0):
            pred = vae_model(data[0])
            estInds = pred.argmax(axis=1)
            estDOAs = classes[estInds]

            trueInds = data[1].argmax(axis=1)
            trueDOAs= classes_off[trueInds]

            error += torch.sum(torch.abs(estDOAs-trueDOAs))
            nData += len(trueDOAs)

    mae = error/nData

    return mae
