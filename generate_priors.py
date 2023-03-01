import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import glob
from noncart_training.trajectory import *



def load_kspace_from_file(fname):
    kspace = np.load(fname)
    if kspace.ndim == 2:
        kspace = kspace[:, :, np.newaxis] # HW => HWC
    kspace = kspace.transpose(2, 0, 1) # HWC => CHW
    return kspace

def complex_to_magphase(self, array_complex):
    array_mag = np.abs(array_complex).astype(np.float32)
    array_phase = (np.angle(array_complex)/np.pi).astype(np.float32) #get phase angle and divide by pi to confine between [-1,+1]
    return np.stack((array_mag,array_phase),ais=0)

def load_raw_image(fname): #Copied from dataset.py dataset class
    kspace_2ch = load_kspace_from_file(fname)
    kspace_complex = kspace_2ch[0,:,:]+kspace_2ch[1,:,:]*1j
    image_complex = sigpy.ifft(kspace_complex)

    points,alpha = generate_trajectory(kspace_complex.shape, interleave_range = (1,8), undersampling = 1, alpha_range = (1,4)) # FIXED TRAJECTORY AT ~1.0 INFORMATION RATIO
    values = interpolate_values(points,kspace_complex)
    
    prior_complex = NUFFT_adjoint(points, values, kspace_complex.shape,alpha)
    prior_complex = prior_complex.astype(np.complex64)
    image_2ch = self.complex_to_magphase(image_complex)
    prior_2ch = self.complex_to_magphase(prior_2ch)

    # squeeze and clip to a range of +-1 within each channel
    # prior_mag = fixed_channelwise_normalization(prior_mag, low=-0.01,high=0.01,clipping=False) #prior magnitude scaling by 1/200
    # image_2ch = fixed_channelwise_normalization(image_2ch, low=-0.001,high=0.001,clipping=False) #image component scaling by 1/2000
    # prior_2ch = fixed_channelwise_normalization(prior_2ch, low=-0.01,high=0.01,clipping=False) #prior component scaling by 1/200
    
    PIL.Image.fromarray((prior_2ch[0,:,:]*127.5+127.5).astype(np.uint8),'L').save('out/prior_real_{}.png'.format(raw_idx))
    PIL.Image.fromarray((prior_2ch[1,:,:]*127.5+127.5).astype(np.uint8),'L').save('out/prior_imag_{}.png'.format(raw_idx))
    PIL.Image.fromarray((image_2ch[0,:,:]*127.5+127.5).astype(np.uint8),'L').save('out/image_real_{}.png'.format(raw_idx))
    PIL.Image.fromarray((image_2ch[1,:,:]*127.5+127.5).astype(np.uint8),'L').save('out/image_imag_{}.png'.format(raw_idx))
    # PIL.Image.fromarray((prior_mag[0,:,:]*127.5+127.5).astype(np.uint8),'L').save('out/prior_magnitude_{}.png'.format(raw_idx))

    # return np.stack((image_2ch, prior_2ch),axis=0)
    return image_2ch, prior_2ch 

# def get_image_and_prior(fname, interleave_range, undersampling, alpha_range):
#     kspace_2ch = load_kspace_from_file(fname)
#     kspace_complex = kspace_2ch[0,:,:]+kspace_2ch[1,:,:]*1j
#     image_complex = sigpy.ifft(kspace_complex)

#     points,alpha = generate_trajectory(kspace_complex.shape, interleave_range = interleave_range, undersampling = undersampling, alpha_range = alpha_range) # FIXED TRAJECTORY AT ~1.0 INFORMATION RATIO
#     values = interpolate_values(points,kspace_complex)
    
#     prior_complex = NUFFT_adjoint(points, values, kspace_complex.shape,alpha)
#     prior_complex = prior_complex.astype(np.complex64)
#     image_2ch = np.stack((image_complex.real, image_complex.imag),axis=0)
#     prior_2ch = np.stack((prior_complex.real, prior_complex.imag),axis=0)

#     return image_2ch, prior_2ch 



@click.command()
@click.option('--kspacedir',             help='Directory containing full kspace arrays', metavar='DIR',        type=str, required=True)
@click.option('--interleave_min',        help='min interleaves', metavar='INT',                                type=int, default=1, show_default=True)
@click.option('--interleave_max',        help='max interleaves', metavar='INT',                                type=int, default=1, show_default=True)
@click.option('--undersampling',         help='undersampling factor (information ratio)', metavar='FLOAT',     type=float, default=0.25, show_default=True)
@click.option('--alpha_min',             help='min alpha', metavar='INT',                                      type=float, default=2, show_default=True)
@click.option('--alpha_max',             help='max alpha', metavar='INT',                                      type=float, default=2, show_default=True)

def main(kspacedir, interleave_min, interleave_max, undersampling, alpha_min, alpha_max):
    #Load ground truth kspace from kspacedir
    kspacelist = sorted(glob.glob(kspacedir+'/*.npy'))
    interleave_range = (interleave_min,interleave_max)
    alpha_range = (alpha_min,alpha_max)
    for kspacefile in tqdm.tqdm(kspacelist):
        image_2ch, prior_2ch = load_raw_image(kspacefile)

        image_magnitude = image_2ch[0][np.newaxis, :, :]
        prior_magnitude = prior_2ch[0][np.newaxis, :, :]

        image_magnitude = (image_magnitude * 100 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        prior_magnitude = (prior_magnitude * 100 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

        outdir = f'{os.path.split(kspacedir)[0]}/il_{interleave_range[0]}-{interleave_range[1]}_u_{int(undersampling*100)}_a_{int(alpha_range[0])}-{int(alpha_range[1])}'
        os.makedirs(outdir, exist_ok=True)

        image_savename = os.path.splitext(os.path.split(kspacefile)[1])[0]+'_magnitude.png'
        prior_mag_savename = os.path.splitext(os.path.split(kspacefile)[1])[0]+'_prior.png'
        prior_2ch_savename = os.path.splitext(os.path.split(kspacefile)[1])[0]+'_prior.npy'

        image_path = os.path.join(kspacedir, image_savename)
        prior_mag_path = os.path.join(outdir, prior_mag_savename)
        prior_2ch_path = os.path.join(outdir, prior_2ch_savename)

        PIL.Image.fromarray(image_magnitude[:, :, 0], 'L').save(image_path)
        PIL.Image.fromarray(prior_magnitude[:, :, 0], 'L').save(prior_mag_path)
        np.save(prior_2ch_path,prior_2ch.astype(np.float32))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
