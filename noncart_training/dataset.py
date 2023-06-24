# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import sigpy
from noncart_training.trajectory import *

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        fetch_raw   = False,    # Return raw kspace data on call (used when sampling)
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self.fetch_raw = fetch_raw

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError
    
    def _load_raw_kspace(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        # raw_data = self._cached_images.get(raw_idx, None)
        # if raw_data is None:
        #     image,prior = self._load_raw_image(raw_idx)
        #     if self._cache:
        #         self._cached_images[raw_idx] = image,prior
        # else:
            # image = raw_data[0]
            # prior = raw_data[1]
        raw_data = self._load_raw_image(raw_idx)
        image = raw_data[0]
        prior = raw_data[1]

        assert isinstance(image, np.ndarray)
        assert isinstance(prior, np.ndarray)

        # assert image.shape == tuple(self.image_shape), f'{image.shape} , {tuple(self.image_shape)}'
        # assert prior.shape == tuple(self.image_shape), f'{prior.shape} , {tuple(self.image_shape)}'

        assert image.dtype == np.float32
        assert prior.dtype == np.float32

        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            assert prior.ndim == 3 # CHW

            image = image[:, :, ::-1]
            prior = prior[:, :, ::-1]

        if self.fetch_raw:
            return torch.tensor(image).to(torch.float32), torch.tensor(prior).to(torch.float32), self.get_label(idx), torch.tensor(self._load_raw_kspace(raw_idx)).flatten(start_dim=0,end_dim=1)
        else:
            return torch.tensor(image).to(torch.float32), torch.tensor(prior).to(torch.float32), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class NonCartesianDataset(Dataset):
    def __init__(self,
        path,                           # Path to directory or zip.
        resolution      = None,         # Ensure specific resolution, None = highest available.
        use_pyspng      = True,         # Use pyspng if available?
        undersampling   = 1,            # Undersampling ratio
        interleaves     = (1,8),        # Interleaves
        alpha_range     = (1,4),        # Alpha range
        **super_kwargs,                 # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self.undersampling = undersampling
        self.interleaves = interleaves
        self.alpha_range = alpha_range

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()

        # Setting list to np array in order to combat multiprocessing leakage
        self._image_fnames = np.array(sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npy')).astype(np.string_)
        # Setting list to torch tensor in order to combat multiprocessing leakage
        #self._image_fnames = torch.tensor(sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npy'))

        if len(self._image_fnames) == 0:
            raise IOError('No numpy array files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0)[0].shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_kspace(self, raw_idx):
        fname = str(self._image_fnames[raw_idx], encoding='utf-8')
        with self._open_file(fname) as f:
            if self._file_ext(fname) == '.npy':
                kspace = np.load(f)
                assert kspace.dtype == np.float32, 'kspace datatype should be float32, half precision results in a significant drop in im quality'
                assert kspace.shape[0] == kspace.shape[1], f'shape kspace = {kspace.shape}, file = {fname}'
            else:
                print('ERROR - tried to load incompatible file type, requires float32 numpy array (.npy)')
                return 0
        if kspace.ndim == 2:
            kspace = kspace[:, :, np.newaxis, np.newaxis] # H,W => H,W,complex,channel
        if kspace.ndim == 3: 
            kspace = kspace[:, :, :, np.newaxis] # H,W,complex => H,W,complex,channel
        kspace = kspace.transpose(3, 2, 0, 1) # H,W,complex,channel => channel,complex,H,W
        return kspace
    
    def complex_to_magphase(self, array_complex):
        magphase = np.empty((array_complex.shape[0]*2, array_complex.shape[1], array_complex.shape[2]), dtype = np.float32)
        magphase[0::2,:,:] = np.abs(array_complex).astype(np.float32)
        magphase[1::2,:,:] = (np.angle(array_complex)/np.pi).astype(np.float32) #get phase angle and divide by pi to confine between [-1,+1]
        return magphase

    def _load_raw_image(self, raw_idx):
        kspace_2ch = self._load_raw_kspace(raw_idx)
        image_complex = np.zeros((kspace_2ch.shape[0],kspace_2ch.shape[2],kspace_2ch.shape[3]),dtype=np.complex64)
        prior_complex = np.zeros((kspace_2ch.shape[0],kspace_2ch.shape[2],kspace_2ch.shape[3]),dtype=np.complex64)
        points,alpha = generate_trajectory(kspace_2ch[0,0,:,:].shape, interleave_range = self.interleaves, undersampling = self.undersampling, alpha_range = self.alpha_range)

        for coil in range(kspace_2ch.shape[0]):
            kspace_complex = kspace_2ch[coil,0,:,:]+kspace_2ch[coil,1,:,:]*1j
            values = interpolate_values(points,kspace_complex) #for complicated interpolation
            # values = map_values(points,kspace_complex) #for simple value mapping
            prior_complex[coil,:,:] = NUFFT_adjoint(points, values, kspace_complex.shape,alpha)
            image_complex[coil,:,:] = sigpy.ifft(kspace_complex)
        del kspace_2ch
        del points

        image_2ch = self.complex_to_magphase(image_complex)
        del image_complex
        prior_2ch = self.complex_to_magphase(prior_complex)
        del prior_complex

        image_2ch = fixed_channelwise_normalization(image_2ch,  low=0,high=0.0025,clipping=False, realonly=True)
        prior_2ch = fixed_channelwise_normalization(prior_2ch,  low=0,high=0.025,clipping=False, realonly=True)

        return np.stack((image_2ch, prior_2ch),axis=0)

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
