import scipy.misc
import multiprocessing as mp
import os 

import sys
sys.path.append('../')

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import sigpy
import pynufft
import skimage
from multiprocessing import Pool
from PIL import Image
import tqdm
import glob

from training.dataset import *
from dnnlib.util import *
from torch_utils import distributed as dist
from torch_utils import misc
import training.trajectory as trajectory
import training.visualize as visualize
import generate_conditional

device = torch.device('cuda')

import torchvision

import time


def save_loop(dataset_iterator, savedir, num_files, subdir_number=10000, repetitions=1):
    print(f'Processing a dataset of {num_files} files times {repetitions} repetitions into subfolders of {subdir_number}')
    for i in range(num_files*repetitions//10000+1):
        remaining = num_files*repetitions - i*subdir_number
        next_n = min(remaining, subdir_number)
        for j in tqdm.tqdm(range(next_n)):
            os.makedirs(f'{savedir}/{str(i).zfill(5)}', exist_ok=True)
            savepath = f'{savedir}/{str(i).zfill(5)}/{str(j).zfill(5)}.pt'
            parcel = next(dataset_iterator)
            torch.save(parcel, savepath)


def main():
    # savedir = '../fastMRIprocessing/more_space/precompute_256_multicoil_T2_test'
    savedir = '../fastMRIprocessing/magnetic_drive/trevor/noncart_dataset/precompute_256_multicoil_T2_train'

    # Load dataset.
    seeds = [0,]
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.NonCartesianDataset', path='../fastMRIprocessing/magnetic_drive/trevor/noncart_dataset/subset2', use_labels=False, xflip=True, fetch_raw=False, undersampling=0.05, interleaves=(4,24), maxiter = 100)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=32, prefetch_factor=3)
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seeds[0])
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=1, **data_loader_kwargs))
	
    save_loop(dataset_iterator, savedir, len(dataset_obj), subdir_number=10000, repetitions=4)
        
if __name__=="__main__":
    main()



