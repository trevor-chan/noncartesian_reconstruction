# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------


@persistence.persistent_class
class UnconditionalEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
    
    # Function to calculate the loss as a weighted sum of magnitude[MSE of magnitude images] and phase[the sum of horizontal and vertical component distance between two phase images]
    def loss_condition(self, yn, y, weight, magphase_b=0.2):
        channels = yn.shape[1]

        yn_mag = yn[:,:channels//2]
        y_mag = y[:,:channels//2]
        yn_pha = yn[:,channels//2:]
        y_pha = y[:,channels//2:]

        hor_n = torch.cos(yn_pha)
        hor = torch.cos(y_pha)
        ver_n = torch.sin(yn_pha)
        ver = torch.sin(y_pha)

        loss_pha = weight * ((hor_n-hor) ** 2 + (ver_n - ver) ** 2) * magphase_b
        loss_mag = weight * ((yn_mag - y_mag) ** 2)

        return loss_pha + loss_mag, loss_mag, loss_pha
    
    def simple_loss_condition(self, yn, y, weight):
        return weight * ((yn - y) ** 2), weight * ((yn - y) ** 2), weight * ((yn - y) ** 2)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        n = torch.randn_like(y) * sigma

        # convert image and prior to magnitude phase representations
        y_mag = torch.abs(y[:,:y.shape[1]//2]+y[:,y.shape[1]//2:]*1j)
        y_pha = torch.angle(y[:,:y.shape[1]//2]+y[:,y.shape[1]//2:]*1j)
        y_magphase = torch.cat((y_mag, y_pha), dim=1)

        # ynp = torch.cat((y+n, prior),dim=1) #concatenate the image and prior in the channel dimension for input into the network as noisy y and prior
        ynp = y_magphase+n

        D_yn = net(ynp, sigma, labels, augment_labels=augment_labels)

        # loss, loss_mag, loss_pha = self.loss_condition(D_yn, y, weight)
        loss, loss_mag, loss_pha = self.loss_condition(D_yn, y_magphase, weight)

        return loss, loss_mag, loss_pha

#----------------------------------------------------------------------------
