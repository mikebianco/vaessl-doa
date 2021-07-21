"""
Code for source localization with conventional DOA estimation. Implemented using
tht Pyroomacoustics library

The method implemented here is described in:
1. M.J. Bianco, S. Gannot, E. Fernandez-Grande, P. Gerstoft, "Semi-supervised
source localization in reverberant environments," IEEE Access, Vol. 9, 2021.
DOI: 10.1109/ACCESS.2021.3087697
2. R.Scheibler, E.Bezzam, I.Dokmanic, "Pyroomacoustics: A Python package for
audio room simulation and array processing algorithms," in Proc. IEEE Int. Conf.
Acoust., Speech Signal Process. (ICASSP), Apr. 2018, pp. 351–355.

If you find this code usefult for your research, please cite these papers.
Michael J. Bianco, July 2021
mbianco@ucsd.edu
"""

import numpy as np
import argparse
import json
import time
import utils.data_cls as data_cls
import pyroomacoustics as pra


def main(config,args):

    PATH = args.eval_data
    r = config["mic locs"]
    my_DOAs = config["doa grid"]

    data_obj = data_cls.DataClass(path=PATH,addNoise=True,noiseSeed=0)

    frames1, frames2, labels = data_obj.get_conventional_data(nframes=args.num_frames,nLabels=args.num_labels) 
    # instantiating doa algorithms
    doa = pra.doa.algorithms[args.algorithm](r,data_obj._fs,data_obj._nfft,c=343,num_src=1,azimuth=my_DOAs,mode='near') 

    X = np.zeros((2,data_obj._nBins,frames1.shape[1])).astype('complex')

    nCases=len(labels)

    out=[]
    target=[]

    start = time.time()

    for case in range(nCases):

        X[0,:,:]=frames1[case,:,:].T
        X[1,:,:]=frames2[case,:,:].T

        doa.locate_sources(X,freq_range=[0,data_obj._fs/2]) # TODO: include nBins calculation here

        est = doa.azimuth_recon 

        out.append(est)
        target.append(labels[case])

    out = np.array(out)*180/np.pi
    target = np.array(target)

    avg_time = (time.time()-start)/nCases

    mae,acc = get_mae_acc(out,target)
    print("mae: ",mae," acc: ",acc,"-",args.algorithm,"-",args.eval_data, "-avg time", avg_time)


def get_mae_acc(estimates,targets):
    error=np.round(estimates.flatten())-targets.flatten()
    mae = np.sum(np.abs(error))/len(error)
    acc = np.sum(np.abs(error)<.0001)/len(error)

    return mae, acc


if __name__ == '__main__':
    with open('default_paths.json') as f:
        default_dict = json.load(f)

    parser = argparse.ArgumentParser(description="Conventional DOA estimation")

    parser.add_argument('-ed','--eval-data', default = default_dict['path_data']+default_dict['data_valid'])
    parser.add_argument('-nl','--num-labels',default = 100, type=int)
    parser.add_argument('-nf','--num-frames',default = 31, type=int)
    parser.add_argument('-algo','--algorithm',default = 'SRP') # alternative: MUSIC

    args = parser.parse_args()

    config = {
        "mic locs": np.array([[0,0],[0,0.085],[0,0]]),
        "doa grid": np.arange(-90,100,10)*np.pi/180, # putting DOAs on exact grid
    }
    main(config,args)
