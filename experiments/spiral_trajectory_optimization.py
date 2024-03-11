# Dataset testing

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import sigpy
import pynufft
import skimage
import os
import multiprocessing as mp
from multiprocessing import Pool
from PIL import Image
import tqdm
import glob
import seaborn as sns

from training.dataset import *
from dnnlib.util import *
from torch_utils import distributed as dist
from torch_utils import misc
import training.trajectory as trajectory
import training.visualize as visualize
import generate_conditional

device = torch.device('cuda')

import torch
import torchvision

import time

import spiral_trajectory_optimization

from skimage.metrics import structural_similarity as ssim

def chs(
    net, latents, priors, kspace, class_labels=None, randn_like=torch.randn_like,
    num_steps=50, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    to_yield=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Convert latents to magnitude phase format
    prior_mag = torch.abs(priors[:,:priors.shape[1]//2]+priors[:,priors.shape[1]//2:]*1j)
    prior_pha = torch.angle(priors[:,:priors.shape[1]//2]+priors[:,priors.shape[1]//2:]*1j)
    priors = torch.cat((prior_mag, prior_pha), dim=1)

    # Main sampling loop.
    assert priors.shape == latents.shape, f'Priors {priors.shape} and latents {latents.shape} passed are incompatible shapes'
    priors = priors.to(torch.float64)
    x_next = latents.to(torch.float64) * t_steps[0]

    # prior_copy = np.array(priors[0].cpu())
    # x_next_copy = np.array(x_next[0].cpu())

    # PIL.Image.fromarray((np.abs(prior_copy[0,:,:]+prior_copy[1,:,:]*1j)).astype(np.uint8),'L').save('out/prior_test.png')
    # PIL.Image.fromarray(np.abs(x_next_copy[0,:,:]+x_next_copy[1,:,:]*1j).astype(np.uint8)*10,'L').save('out/x_next_test.png')


    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #Concatenate x_hat and priors (undersampled 2ch images) for first forward pass
        
        x_in = torch.cat((x_hat, priors), dim=1)

        # Euler step.
        denoised = net(x_in, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            #Concatenate x_hat and priors (undersampled 2ch images) for forward pass
            x_in = torch.cat((x_next, priors), dim=1)

            denoised = net(x_in, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
        yield({'x': x_next, 'denoised': denoised,})
    
    # convert back to real imaginary representation
    x_next = trajectory.magphase_to_complex(x_next)

    return x_next


def make_trajectory_custom(matsize, undersampling=1, interleaves=1, alpha=1, freqcrop = True):

    fov = .22 #in meters
    # adjustedshape = np.power(matsize[0]**2+matsize[1]**2,0.5)
    adjustedshape = matsize[0]
    frequency_encode_undersampling = 1
    max_gradient_amp = 0.045 / 2#T/m  4.5 G/cm, or 2.2 G/cm       - Brian uses 0.040
    max_slew_rate = 0.2 * 1000 #T/m/s

    # u = .0365 / undersampling * (1-2*(((1/alpha)**4)-(1/alpha)**2)) # Empirically determined as a decent approximation for maintaining a steady effective undersampling rate
    # print(u)
    # u = u*10
    u = undersampling
    


    # points = sigpy.mri.spiral(fov, 
    #                         adjustedshape, 
    #                         frequency_encode_undersampling, 
    #                         u, 
    #                         interleaves, 
    #                         alpha, 
    #                         max_gradient_amp, 
    #                         max_slew_rate)
    
    points, t, dt, Tend, Dt = spiral(fov, 
                            adjustedshape, 
                            frequency_encode_undersampling, 
                            u, 
                            interleaves, 
                            alpha, 
                            max_gradient_amp, 
                            max_slew_rate)
    
    if freqcrop:
        precrop_points = points.shape[0]
        points = np.delete(points, np.where((np.hypot(points[:,0],points[:,1]) >= matsize[0]/2)), axis=0) # circle
        # points = np.asarray([i for i in points if -matsize[0]/2<=i[0]<=matsize[0]/2 and -matsize[1]/2<=i[1]<=matsize[1]/2]) # square 
        postcrop_points = points.shape[0]
        crop_accel_factor = postcrop_points/precrop_points
        Tend = Tend*crop_accel_factor

    return points, t, dt, Tend, Dt


def spiral(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm, gamma=2.678e8):
    """Generate variable density spiral trajectory.

    Args:
        fov (float): field of view in meters.
        N (int): effective matrix shape.
        f_sampling (float): undersampling factor in freq encoding direction.
        R (float): undersampling factor.
        ninterleaves (int): number of spiral interleaves
        alpha (float): variable density factor
        gm (float): maximum gradient amplitude (T/m)
        sm (float): maximum slew rate (T/m/s)
        gamma (float): gyromagnetic ratio in rad/T/s

    Returns:
        array: spiral coordinates.

    References:
        Dong-hyun Kim, Elfar Adalsteinsson, and Daniel M. Spielman.
        'Simple Analytic Variable Density Spiral Design.' MRM 2003.

    """
    
    res = fov / N

    lam = 0.5 / res  # in m**(-1)
    n = 1 / (1 - (1 - ninterleaves * R / fov / lam) ** (1 / alpha))
    w = 2 * np.pi * n
    Tea = lam * w / gamma / gm / (alpha + 1)  # in s
    Tes = np.sqrt(lam * w**2 / sm / gamma) / (alpha / 2 + 1)  # in s
    Ts2a = (
        Tes ** ((alpha + 1) / (alpha / 2 + 1))
        * (alpha / 2 + 1)
        / Tea
        / (alpha + 1)
    ) ** (
        1 + 2 / alpha
    )  # in s

    if Ts2a < Tes:
        tautrans = (Ts2a / Tes) ** (1 / (alpha / 2 + 1))

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (
                t <= Ts2a
            ) + ((t - Ts2a) / Tea + tautrans ** (alpha + 1)) ** (
                1 / (alpha + 1)
            ) * (
                t > Ts2a
            ) * (
                t <= Tea
            ) * (
                Tes >= Ts2a
            )

        Tend = Tea
    else:

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= Tes)

        Tend = Tes

    def k(t):
        return lam * tau(t) ** alpha * np.exp(w * tau(t) * 1j)

    dt = Tea * 1e-4  # in s

    Dt = dt * f_sampling / fov / abs(k(Tea) - k(Tea - dt))  # in s

    t = np.linspace(0, Tend, int(Tend / Dt))
    kt = k(t)  # in rad

    # generating cloned interleaves
    k = kt
    for i in range(1, ninterleaves):
        k = np.hstack((k, kt[0:] * np.exp(2 * np.pi * 1j * i / ninterleaves)))

    k = np.stack((np.real(k), np.imag(k)), axis=1)

    return k, t, dt, Tend, Dt

def parameterize_acceleration_factor(accel_target = 20, interleaves = 4, alpha = 1.2, matsize = 256, reference_time = 112, threshold=0.001, correction = 1, freqcrop = True):
    accel = accel_target*(threshold+1.1)
    undersampling = 1 * matsize / 256 * correction
    iter = 0
    while abs(accel_target-accel) > (threshold * accel_target):
        points, t, dt, Tend, Dt = make_trajectory_custom((matsize,matsize), undersampling=undersampling, interleaves=interleaves, alpha=alpha, freqcrop = freqcrop)
        
        if accel_target-accel > 0:
            undersampling += 0.01 * np.clip(accel_target-accel,0.0001,0.4) * (1000/(1000+iter))
        else:
            undersampling += 0.01 * np.clip(accel_target-accel,-0.0001,-0.4) * (1000/(1000+iter))
                        
        accel = reference_time / (Tend*interleaves)
        iter += 1
        
        # print(f'Undersampling: {undersampling}, Acceleration: {accel}')
        
    # print(f'{iter} iterations')
    return undersampling, interleaves, alpha, accel

def make_trajectory_fixed_accel(matsize, reference_time, acceleration=1, interleaves=1, alpha=1, threshold = 0.01, freqcrop = True):
    undersampling, _, _, _ = parameterize_acceleration_factor(accel_target = acceleration, interleaves = interleaves, alpha = alpha, matsize = matsize, reference_time = reference_time, threshold=threshold, freqcrop = freqcrop)
    points, _, _, Tend, Dt = make_trajectory_custom((matsize,matsize), undersampling=undersampling, interleaves=interleaves, alpha=alpha, freqcrop = freqcrop)
    # print(f'Undersampling trajectory input: {undersampling}')
    return points, Dt, interleaves, alpha, Tend

def plot_trajectory_gradients(points, Dt, interleaves, alpha, Tend):
    import seaborn as sns
    # sns.set_palette('Greens_r')
    sns.set_palette('Reds_r')
    # sns.set_palette('Spectral')

    fig, ax = plt.subplots(1,1,figsize=(6,2))
    if interleaves == 1:
        width = 0.1
    else:
        width = 0.3
    # extent = 4800
    ax.plot(Dt*np.array(list(range(points.shape[0])[:points.shape[0]//interleaves])),points[:points.shape[0]//interleaves], linewidth=width, alpha=1)
    ax.plot(Dt*np.array(list(range(points.shape[0]))),points, linewidth=0.1, alpha=1)

    # ax.plot(Dt*np.array(list(range(points.shape[0]))),points[:,1], linewidth=0.2, alpha=1, color='orange')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('kx, ky (1/m)')
    ax.text(0.995, 0.19, 'Interleaves: {}\nAlpha:   {}\nGradient Amplitude Cap: {} T/m\nGradient Slew Rate Cap: {} T/m/ms\nFOV: {} mm\nTime Interleave: {} ms\n'.format(interleaves, alpha, 0.045, 0.2, 220, str(round(Tend * 1000, 3))), 
            horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, size=8)
    
    
def plot_trajectory_spiral(points, interleaves):
    if interleaves == 1:
        width = 0.1
    else:
        width = 0.3
    g = sns.JointGrid(height=5)
    sns.lineplot(x=points[:points.shape[0]//interleaves,0],y=points[:points.shape[0]//interleaves,1], ax=g.ax_joint, sort=False, alpha=1, linewidth=width)
    for i in range(1,interleaves):
        sns.lineplot(x=points[points.shape[0]//interleaves * i:points.shape[0]//interleaves * (i+1),0],y=points[points.shape[0]//interleaves * i:points.shape[0]//interleaves * (i+1),1],
                    ax=g.ax_joint, sort=False, alpha=1, linewidth=0.1)
    sns.histplot(x=points[:,0], alpha = 0.7, linewidth=0.1, ax=g.ax_marg_x)
    sns.histplot(y=points[:,1], alpha = 0.7, linewidth=0.1, ax=g.ax_marg_y)
    # sns.kdeplot(x=points[:points.shape[0]//interleaves,0],y=points[:points.shape[0]//interleaves,1], color="r", alpha=.1, levels=10, ax=g.ax_joint)
    # sns.scatterplot(x=points[:points.shape[0]//interleaves,0],y=points[:points.shape[0]//interleaves,1], color="b", marker = '.', s=10, alpha=0.5, ax=g.ax_joint, zorder=100)

    g.ax_joint.set_xlabel('kx (1/m)')
    g.ax_joint.set_ylabel('ky (1/m)')
    
    
def load_raw_kspace(fname):
    with open(fname, 'rb') as f:
        kspace = np.load(f)
        assert kspace.dtype == np.float32, 'kspace datatype should be float32, half precision results in a significant drop in im quality'
        assert kspace.shape[0] == kspace.shape[1], f'shape kspace = {kspace.shape}, file = {fname}'
    if kspace.ndim == 2:
        kspace = kspace[:, :, np.newaxis, np.newaxis] # H,W => H,W,complex,channel
    if kspace.ndim == 3: 
        kspace = kspace[:, :, :, np.newaxis] # H,W,complex => H,W,complex,channel
    kspace = kspace.transpose(3, 2, 0, 1) # H,W,complex,channel => channel,complex,H,W
    kspace = kspace[:,0] + kspace[:,1] * 1j # convert to [channel, h, w] as complex number
    
    return kspace

def load_raw_image(fname, accel_target, interleaves, alpha, matsize, reference_time):
    
    kspace = load_raw_kspace(fname)    
    points, Dt, interleaves, alpha, Tend = make_trajectory_fixed_accel(matsize, reference_time, acceleration=accel_target, interleaves=interleaves, alpha=alpha,  threshold=0.01)

    # initialize nufftobj
    nufftobj = trajectory.prealloc_nufft(kspace, points)

    # compute the values via complex interpolation, compute the image prior via inverse nufft, and compute the ground truth image
    y = trajectory.interpolate_values(points,kspace) #for complex interpolation
    prior = trajectory.inverse_nufft(y, nufftobj, maxiter=100)
    image = sigpy.ifft(kspace, axes=(-1,-2))

    # perform std normalization
    prior = trajectory.intensity_normalization(prior)
    image = trajectory.intensity_normalization(image)

    kspace = {'kspace':torch.tensor(kspace).unsqueeze(0), 'y':torch.tensor(y).unsqueeze(0), 'points':torch.tensor(points).unsqueeze(0)}

    image_tensor = trajectory.complex_to_float(image)
    prior_tensor = trajectory.complex_to_float(prior)
        
    return image_tensor.unsqueeze(0), prior_tensor.unsqueeze(0), 0, kspace

def evaluate_trajectory(net, parcel, config, seed=0):
    torch.manual_seed(seed)
    
    batch = parcel[0].shape[0]
    latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    priors = parcel[1].to(device)
    kspace = parcel[3]['kspace'].to(device)

    y = np.array(parcel[3]['y'])
    points = np.array(parcel[3]['points'])
    
    # No guidance
    for s in chs(net=net, latents=latents, priors=priors, kspace=kspace, num_steps=config['num_steps'], sigma_min=config['sigma_min'], sigma_max=config['sigma_max']):
        pass
    recon = trajectory.magphase_to_complex(s['denoised'])
    image_rss = trajectory.root_summed_squares(trajectory.float_to_complex(parcel[0]))[0]
    recon_rss = trajectory.root_summed_squares(recon)[0]
    return ssim(image_rss, recon_rss, data_range=np.amax(image_rss) - np.amin(image_rss))

def aggregate_results(datas, net, accel_target, interleaves, alphas, matsize, reference_time, config):
    ssims = np.zeros((len(interleaves), len(alphas)))
    for i, interleave in tqdm.tqdm(enumerate(interleaves), total=interleaves.shape[0]):
        for j, alpha in enumerate(alphas):
            ssim = 0
            for data in datas:
                parcel = load_raw_image(data, accel_target, interleave, alpha, matsize, reference_time)
                ssim += evaluate_trajectory(net, parcel, config, seed=0)
            ssims[i, j] = ssim / len(datas)
            # print(ssims[i, j])
    return ssims