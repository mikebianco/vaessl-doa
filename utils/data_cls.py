"""
Data class for semi-supervised variational autoencoder (VAE-SSL) model

The method implemented here is described in:
1. M.J. Bianco, S. Gannot, E. Fernandez-Grande, P. Gerstoft, "Semi-supervised
source localization in reverberant environments," IEEE Access, Vol. 9, 2021.
DOI: 10.1109/ACCESS.2021.3087697

If you find this code usefult for your research, please cite (1)
Michael J. Bianco, July 2021
mbianco@ucsd.edu
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy.random as nprand


class DataClass:
    def __init__(self, path='', nfft=256, addNoise=False, noiseDB = 20, noiseSeed = 0, cuda_id = None, cnn = False,
                        num_workers=0, loader_shuffle = True, norm_factor = np.pi, fs=16e3):

        self._nfft=nfft
        self._path = path
        self._seed = noiseSeed
        self._cuda_id = cuda_id
        self._num_workers = num_workers
        self._cnn = cnn
        self._loader_shuffle = loader_shuffle
        self._norm_factor = norm_factor
        self._fs = fs
        self._use_cuda = cuda_id is not None


        # loading data
        data0=sio.loadmat(self._path)

        data=data0['data']
        nSeq=data.shape[0]
        spectral_feat=Spectrogram(nfft=self._nfft)


        # extracting rtf features
        doas=[]
        RTFs=[]
        specs=[]
        nSeq=data.shape[0]
        nprand.seed(self._seed)   # pseudo-random noise

        # calcualting noise fraction based on dB
        noiseFrac = 10**(-noiseDB/20)
        print('noise frac:', noiseFrac)


        for n in range(nSeq):
            seq=data[n,1]
            doa=data[n,0]
            seq0 = seq[:,0]
            seq1 = seq[:,1]

            if addNoise == True:

                seq0+=noiseFrac*seq0.std()*nprand.randn(len(seq0))
                seq1+=noiseFrac*seq1.std()*nprand.randn(len(seq1))

            spec0=np.squeeze(spectral_feat._spectrogram(seq0[500:],windowing=True))
            spec1=np.squeeze(spectral_feat._spectrogram(seq1[500:],windowing=True))

            specs_out=np.concatenate((spec0,spec1),axis=1)

            doas.extend((doa*np.ones((specs_out.shape[0],))).squeeze().tolist())
            specs.extend(specs_out.tolist())

        doas  = np.array(doas)
        doas  = doas.reshape((-1,1))

        specs = np.array(specs)

        doas.flags.writeable  = False
        specs.flags.writeable = False

        self._doas_array  = doas
        self._specs_array = specs

        self._rtfs_array = self._specs_array

        self._label_set = list(set(self._doas_array.flatten())).copy()


    def get_train_test(self, frame_dict, inds):
        frames_out = []
        labels_out = []
        for k in inds:  # train/test indices
            for l in frame_dict.keys(): # looping over labels
                frames_out.append(frame_dict[l][k])
                labels_out.append([l])

        return np.array(frames_out), np.array(labels_out)


    def get_stft_seq(self, nframes=10,nLabels=100, plotting=False, import_stat=(None,None)):
        self._nFrames = nframes

        # checking for good/bad frames
        rtf_frames = []
        rtf_labels = []
        frame_check =[]
        for k in range(len(self._rtfs_array)-nframes+1):
            rtf_frames0 = self._rtfs_array[k:k+nframes,:]
            rtf_labels0 = self._doas_array[k:k+nframes]
            rtf_frames.append(rtf_frames0)

            if np.sum(np.abs(rtf_labels0-rtf_labels0[0])) == 0:
                rtf_labels.extend(rtf_labels0[0])
                frame_check.append(0)
            else:
                rtf_labels.append(rtf_labels0[0])
                frame_check.append(1)

        rtf_labels  = np.array(rtf_labels)
        rtf_frames  = np.array(rtf_frames)
        frame_check = np.array(frame_check)

        rtf_labels.flags.writeable  = False
        rtf_frames.flags.writeable  = False
        frame_check.flags.writeable  = False

        self._rtf_labels = rtf_labels
        self._rtf_frames  = rtf_frames
        self._frame_check = frame_check

        rtf_frames_norm = self._rtf_frames


        del self._rtf_frames

        rtf_frames_norm.flags.writeable  = False

        self._rtf_frames_norm = rtf_frames_norm

        # addressing data imbalance
        rtf_frames_good = self._rtf_frames_norm[self._frame_check==0,:,:]
        rtf_labels_good = self._rtf_labels[self._frame_check==0]

        rtf_frames_bad = self._rtf_frames_norm[self._frame_check==1,:,:]
        rtf_labels_bad = self._rtf_labels[self._frame_check==1].reshape(-1,1)

        del self._rtf_frames_norm



        if not self._loader_shuffle:
            ##### shuffling frames, gives deterministic shuffle from output of data_loaders
            nprand.seed(0)
            shuffle_inds = nprand.permutation(rtf_frames_good.shape[0])

            self._rtf_frames_good = rtf_frames_good[shuffle_inds,:,:].copy()
            self._rtf_labels_good = rtf_labels_good[shuffle_inds].copy()

        else:
            ##### not shuffling frames
            rtf_frames_good.flags.writeable  = False
            rtf_labels_good.flags.writeable  = False

            self._rtf_frames_good = rtf_frames_good # looks good up to here
            self._rtf_labels_good = rtf_labels_good

        a=set(self._rtf_labels_good)

        df = {k: [] for k in a}

        for k in range(len(self._rtf_labels_good)):
            df[self._rtf_labels_good[k]].append(self._rtf_frames_good[k,:,:])

        lmin = float('inf')


        for k in df.keys():  lmin = min(lmin,len(df[k]))  # smallest number of examples in all keys (DOAs)

        assert nLabels >= len(a), "Too few labels! Number of labels must be >= number of DOAs."

        nAllow=(nLabels//len(a))

        train_inds = np.linspace(0,lmin-1,nAllow).astype(int)
        train_inds_s = set(train_inds)
        test_inds_s0 = set(np.arange(0,lmin-1))
        test_inds_s = test_inds_s0.difference(train_inds_s)
        test_inds = np.array(list(test_inds_s))

        self._train_inds = train_inds
        self._test_inds = test_inds

        frame_train, label_train = self.get_train_test(df,train_inds)
        frame_test,  label_test  = self.get_train_test(df,test_inds)

        if plotting == True:
            plt.figure(0)
            plt.hist(label_train,100)
            plt.title('labeled data distribution')

            plt.figure(1)
            plt.hist(label_test,100)
            plt.title('unlabeled distribution')

        print('nlabels:', len(label_train))


        del df # removing duplication

        frame_dict = {}
        frame_dict['train'] = frame_train
        frame_dict['test'] = frame_test
        frame_dict['bad'] = rtf_frames_bad

        label_dict = {}
        label_dict['train'] = label_train
        label_dict['test'] = label_test
        label_dict['bad'] = rtf_labels_bad

        return frame_dict, label_dict



    def get_rtf_phase(self, specs, nBins = 127, norm_factor = np.pi):
        self._nBins = nBins
        len_specs = specs.shape[-1]//2
        assert nBins <= len_specs, "Too many frequency bins! Must be < nfft/2+1"
        spec0 = specs[:,:,:len_specs]
        spec1 = specs[:,:,len_specs:]
        Y2Y1=spec1*spec0.conj()
        Y1Y1=spec0*spec0.conj()

        RTF_phase = np.angle(Y2Y1/Y1Y1)[:,:,:nBins]
        RTF_phase /= norm_factor

        return RTF_phase


    def get_vaessl_data(self, nframes=10, nLabels=100, plotting=False, batch_size = 256, nBins =127):
        frame_dict, label_dict = self.get_stft_seq(nframes=nframes, nLabels=nLabels, plotting=plotting)
        for key in frame_dict.keys():
            frame_dict[key] = self.get_rtf_phase(frame_dict[key], nBins=nBins)  # overwriting frame_dict
            print(key,'frames shape:', frame_dict[key].shape)

        dataset_sup = DOA_RTF_Dataset(frame_dict['train'],label_dict['train'],self._label_set,self._use_cuda,self._cuda_id)
        dataset_valid_test = DOA_RTF_Dataset(frame_dict['test'],label_dict['test'],self._label_set,self._use_cuda,self._cuda_id) # excluding bad frames

        frame_unsup = np.concatenate((frame_dict['bad'],frame_dict['test']),axis=0)
        label_unsup = np.concatenate((label_dict['bad'],label_dict['test']),axis=0)

        dataset_unsup = DOA_RTF_Dataset(frame_unsup,label_unsup,self._label_set,self._use_cuda,self._cuda_id)

        data_loaders={}

        data_loaders["sup"]   = DataLoader(dataset_sup, batch_size=batch_size,shuffle=self._loader_shuffle, num_workers=self._num_workers)
        data_loaders["unsup"] = DataLoader(dataset_unsup, batch_size=batch_size,shuffle=self._loader_shuffle, num_workers=self._num_workers)
        data_loaders["test"]  = DataLoader(dataset_valid_test, batch_size=batch_size,shuffle=self._loader_shuffle, num_workers=self._num_workers)

        return data_loaders


    def get_cnn_data(self, nframes=10, nLabels=100, plotting=False, batch_size = 256, nBins =127):
        self._nBins = nBins
        frame_dict, label_dict = self.get_stft_seq(nframes=nframes, nLabels=nLabels, plotting=plotting)
        for key in frame_dict.keys():
            frame_dict[key] = self.get_rtf_phase(frame_dict[key], nBins=nBins)  # overwriting frame_dict
            print(key,'frames shape:', frame_dict[key].shape)

        dataset_sup = CNN_Dataset(frame_dict['train'],label_dict['train'],self._label_set,self._use_cuda,self._cuda_id)
        dataset_valid_test = CNN_Dataset(frame_dict['test'],label_dict['test'],self._label_set,self._use_cuda,self._cuda_id) # excluding bad frames

        data_loaders={}

        data_loaders["sup"]   = DataLoader(dataset_sup, batch_size=batch_size,shuffle=self._loader_shuffle, num_workers=self._num_workers)
        data_loaders["test"]  = DataLoader(dataset_valid_test, batch_size=batch_size,shuffle=self._loader_shuffle, num_workers=self._num_workers)

        return data_loaders


    def get_conventional_data(self, nframes=10, nBins=None, nLabels = 19):

        frame_dict, label_dict = self.get_stft_seq(plotting=True, nframes=nframes, nLabels=nLabels)

        frames = frame_dict['test']
        labels = label_dict['test']

        print('frames shape:', frames.shape)


        len_specs = frames.shape[-1]//2

        if nBins:
            self._nBins = nBins
            assert nBins <= len_specs, "Too many frequency bins! Must be < nfft/2+1"
            frames1 = frames[:,:,:nBins]
            frames2 = frames[:,:,len_specs:len_specs+nBins]
        else:
            self._nBins = len_specs
            frames1 = frames[:,:,:len_specs]
            frames2 = frames[:,:,len_specs:]

        return frames1, frames2, labels


# inheriting pytorch Dataset class, and modifying requisite key methods
class DOA_RTF_Dataset(Dataset):
    """DOA_RTF_Dataset."""

    def __init__(self, rtf_array, rtf_labels, label_set, use_cuda, cuda_id):

        self._rtf_array = rtf_array  # phase of rtf
        self._labels = rtf_labels
        self._label_set = label_set
        self._cuda_id = cuda_id
        self._use_cuda = use_cuda

    def __len__(self):
        return len(self._rtf_array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        lab_one_hot = np.zeros(len(self._label_set))
        hot_idx = self._label_set.index(self._labels[idx])
        lab_one_hot[hot_idx]=1

        rtf_frames = self._rtf_array[idx,:,:].flatten()  # flattening the rtf frames
        rtf_frames = rtf_frames.astype('float32')
        rtf_labels = lab_one_hot
        rtf_labels = rtf_labels.astype('float32')

        if self._use_cuda:
            return torch.tensor(rtf_frames).cuda(self._cuda_id),torch.tensor(rtf_labels).cuda(self._cuda_id)
        else:
            return torch.tensor(rtf_frames),torch.tensor(rtf_labels)

# inheriting pytorch Dataset class, and modifying requisite key methods
class CNN_Dataset(Dataset):
    """DOA_RTF_Dataset."""

    def __init__(self, rtf_array, rtf_labels, label_set, use_cuda, cuda_id):

        self._rtf_array = rtf_array  # phase of rtf
        self._labels = rtf_labels
        self._label_set = label_set #list(set(rtf_labels.flatten()))
        self._cuda_id = cuda_id
        self._use_cuda = use_cuda

    def __len__(self):
        return len(self._rtf_array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        doa_idx = self._label_set.index(self._labels[idx])

        rtf_frames = self._rtf_array[idx,None,:,:] # adding singleton dimension for CNN
        rtf_frames = rtf_frames.astype('float32')
        rtf_labels = doa_idx

        if self._use_cuda:
            return torch.from_numpy(rtf_frames).cuda(self._cuda_id), torch.tensor(rtf_labels).long().cuda(self._cuda_id)
        else:
            return torch.from_numpy(rtf_frames), torch.tensor(rtf_labels).long()


class Spectrogram:
    def __init__(self, nfft=512,nb_channels=1):

        self._nfft = nfft
        self._win_len = self._nfft
        self._hop_len = self._nfft//2 # 50% of frame overlap, mjb used to be 1 (most overlap)
        self._nb_channels = nb_channels

    def _spectrogram(self, audio_input,windowing=False):
        self._audio_max_len_samples = audio_input.shape[0]

        self._max_frames = int(np.ceil((self._audio_max_len_samples - self._win_len) // float(self._hop_len)))

        if windowing == True:
            hann_win = np.repeat(np.hanning(self._win_len)[np.newaxis].T, self._nb_channels)  #mjb
        else:
            hann_win = np.repeat(np.ones(self._win_len)[np.newaxis].T, self._nb_channels)

        nb_bins = self._nfft // 2 +1
        spectra = np.zeros((self._max_frames, nb_bins, self._nb_channels), dtype=complex)
        for ind in range(self._max_frames):
            start_ind = ind * self._hop_len
            aud_frame = audio_input[start_ind + np.arange(0, self._win_len)] * hann_win
            spectra0 = np.fft.fft(aud_frame, n=self._nfft, axis=0, norm='ortho')[:nb_bins]  # one dimension, mjb
            spectra[ind]=spectra0.reshape(-1,1)
        return spectra
