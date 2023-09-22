# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

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
from torch_utils import misc
import glob
import matplotlib.pyplot as plt
from training.trajectory import *


#----------------------------------------------------------------------------
# Modified conditional EDM sampler

def conditional_huen_sampler(
    net, latents, priors, kspace, class_labels=None, randn_like=torch.randn_like,
    num_steps=50, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    to_yield=False,
):
    
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
        
        # convert back to real imaginary representation
        x_next = magphase_to_complex(x_next)
        
        return x_next

    def chs_yield(
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

        # Main sampling loop.
        assert priors.shape == latents.shape, f'Priors {priors.shape} and latents {latents.shape} passed are incompatible shapes'
        priors = priors.to(torch.float64)
        x_next = latents.to(torch.float64) * t_steps[0]

        prior_copy = np.array(priors[0].cpu())
        x_next_copy = np.array(x_next[0].cpu())

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
            
            if to_yield:
                yield dnnlib.EasyDict(x=x_next, denoised=denoised, step=i+1, num_steps=num_steps, t=t_next, c=0, noise_std=t_next)

        return x_next
    
    if to_yield:
        return chs_yield(net, latents, priors, kspace, class_labels=None, randn_like=torch.randn_like,
                         num_steps=50, sigma_min=0.002, sigma_max=80, rho=7,
                         S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,)
    else:
        return chs(net, latents, priors, kspace, class_labels=None, randn_like=torch.randn_like,
                   num_steps=50, sigma_min=0.002, sigma_max=80, rho=7,
                   S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,)

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

    # Main sampling loop.
    assert priors.shape == latents.shape, f'Priors {priors.shape} and latents {latents.shape} passed are incompatible shapes'
    priors = priors.to(torch.float64)
    x_next = latents.to(torch.float64) * t_steps[0]

    prior_copy = np.array(priors[0].cpu())
    x_next_copy = np.array(x_next[0].cpu())

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
        
        if to_yield:
            yield dnnlib.EasyDict(x=x_next, denoised=denoised, step=i+1, num_steps=num_steps, t=t_next, c=0, noise_std=t_next)

    return x_next

def chs_yield(
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

    # Main sampling loop.
    assert priors.shape == latents.shape, f'Priors {priors.shape} and latents {latents.shape} passed are incompatible shapes'
    priors = priors.to(torch.float64)
    x_next = latents.to(torch.float64) * t_steps[0]

    prior_copy = np.array(priors[0].cpu())
    x_next_copy = np.array(x_next[0].cpu())

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
        
        if to_yield:
            yield dnnlib.EasyDict(x=x_next, denoised=denoised, step=i+1, num_steps=num_steps, t=t_next, c=0, noise_std=t_next)

    return x_next


#Proposed frequency-aware 2nd order sampler
def kspace_gradient_sampler(
    net, latents, priors, kspace, class_labels=None, randn_like=torch.randn_like,
    num_steps=50, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    guidance = 0.00001, to_yield=False
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    assert priors.shape == latents.shape, f'Priors {priors.shape} and latents {latents.shape} passed are incompatible shapes'
    priors = priors.to(torch.float64)
    x_next = latents.to(torch.float64) * t_steps[0]

    # compute discrete complex kspace matrix
    k_complex = kspace[:,::2] + kspace[:,1::2]*1j
    
    

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

        # Calculate frequency space gradient
        x_next_complex = x_next[:,::2] * (torch.cos(x_next[:,1::2]*torch.pi) + torch.sin(x_next[:,1::2]*torch.pi) * 1j)
        k_next_complex = torch.fft.fft2(x_next_complex)

        # k_next = torch.cat((torch.zeros_like(k_next_complex),torch.zeros_like(k_next_complex)), dim=1)
        # k_next[:,::2] = k_next_complex.real
        # k_next[:,1::2] = k_next_complex.imag

        k_gradient_complex = k_complex-k_next_complex
        k_gradient_complex[k_complex==0+0j]=0+0j

        x_gradient_complex = torch.fft.fftshift(torch.fft.ifft2(k_gradient_complex))
        x_gradient = torch.cat((torch.zeros_like(x_gradient_complex),torch.zeros_like(x_gradient_complex)), dim=1).to(torch.float32)
        x_gradient[:,::2] = torch.abs(x_gradient_complex).to(torch.float32)
        x_gradient[:,1::2] = (torch.angle(x_gradient_complex)/torch.pi).to(torch.float32)

        x_next = x_next + x_gradient * guidance

        if to_yield:
            yield dnnlib.EasyDict(x=x_next, denoised=denoised, step=i+1, num_steps=num_steps, t=t_next, c=0, noise_std=t_next,
                                  k_gradient_complex=k_gradient_complex, x_gradient=x_gradient)

    return x_next


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        # assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--priordir',                help='Directory containing kspace priors', metavar='DIR',                 type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=False)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=16, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, priordir, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    # Generate 64 images from a directory of priors and save as out/*.png
    python generate_conditional.py \\
        --priordir=fastMRIprocessing/data_small \\
        --outdir=out --seeds=0 --batch=64 \\
        --network=training-runs/00049-data_small-uncond-ddpmpp-edm-gpus1-batch32-fp32/network-snapshot-002500.pkl

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load dataset.
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.NonCartesianDataset', path=priordir, use_labels=False, xflip=False, fetch_raw=True)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=8, prefetch_factor=2)
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seeds[0])
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=max_batch_size, **data_loader_kwargs))

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        latents = latents.repeat(max_batch_size,1,1,1)
        
        #Load and generate priors
        # priorlist = sorted(glob.glob(priordir+'/*.npy'))
        # assert len(priorlist)>=max_batch_size, 'Prior directory contains too few images, add more or decrease --batch'
        # priors = np.stack([np.load(prior) for prior in priorlist[:max_batch_size]],axis=0) #load a batch of priors from the list and stack in batch dim
        # assert priors.shape == latents.shape, f'Priors shape {priors.shape} does not match latents shape {latents.shape}'
        # priors = torch.tensor(priors).to(device)

        images, priors, labels, kspace = next(dataset_iterator)
        images = images.to(device).to(torch.float32)
        priors = priors.to(device).to(torch.float32)
        kspace = kspace.to(device).to(torch.float32)
        labels = labels.to(device)

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        # sampler_fn = conditional_huen_sampler
        sampler_fn = kspace_gradient_sampler
        images = sampler_fn(net, latents, priors, kspace, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Convert from complex to grayscale magnitude images
        images_pha = images[:,1,:,:].cpu().numpy()
        images_mag = images[:,0,:,:].cpu().numpy()

        assert len(images_mag.shape)==3
        assert len(images_pha.shape)==3

        # Save images.
        images_mag_batch = ((images_mag+1)*127.5).clip(0, 255).astype(np.uint8)
        cm = plt.get_cmap('twilight')
        images_pha = (cm((images_pha+1)/2)*2)-1
        images_pha_batch = ((images_pha+1)*127.5).clip(0, 255).astype(np.uint8)
        for idx in range(max_batch_size):
            os.makedirs(outdir, exist_ok=True)
            savename_mag = os.path.splitext(os.path.split(priorlist[idx])[1])[0]+'_mag_recon.png'
            savename_pha = os.path.splitext(os.path.split(priorlist[idx])[1])[0]+'_pha_recon.png'
            image_mag_path = os.path.join(outdir, savename_mag)
            image_pha_path = os.path.join(outdir, savename_pha)
            image_mag = images_mag_batch[idx,:,:]
            image_pha = images_pha_batch[idx,:,:]
            if len(image_mag.shape) == 2:
                PIL.Image.fromarray(image_mag[:, :], 'L').save(image_mag_path)
                PIL.Image.fromarray(image_pha, 'RGB').save(image_pha_path)
            else:
                PIL.Image.fromarray(image_mag, 'RGB').save(image_mag_path)
                PIL.Image.fromarray(image_pha, 'RGB').save(image_pha_path)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
