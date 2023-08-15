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
# Conditional loss function modeled off of:
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class ConditionalEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    # #Function to calculate the distance/loss between two phase images on a -1 to +1 scale
    # def angular_distance(self, yn, y):
    #     top_dist = torch.abs(yn - y)
    #     yn[yn<0]+=1
    #     yn[yn>0]-=1
    #     y[y<0]+=1
    #     y[y>0]-=1
    #     bot_dist = torch.abs(yn - y)
    #     angular_dist = torch.where(top_dist<bot_dist, top_dist, bot_dist)
    #     angular_dist = angular_dist*(1-angular_dist) #reduces magnitude of phase loss in the case where it is 180 degrees off - hopefully resulting in better stability
    #     return angular_dist
    
    # Function to calculate the loss as a weighted sum of magnitude[MSE of magnitude images] and phase[the sum of horizontal and vertical component distance between two phase images]
    def loss_condition(self, yn, y, weight):
        yn_mag = yn[:,::2,:,:]
        y_mag = y[:,::2,:,:]
        yn_pha = yn[:,1::2,:,:]
        y_pha = y[:,1::2,:,:]
        hor_n = torch.cos(yn_pha * torch.pi)
        hor = torch.cos(y_pha * torch.pi)
        ver_n = torch.sin(yn_pha * torch.pi)
        ver = torch.sin(y_pha * torch.pi)
        loss_pha = weight * ((hor_n-hor) ** 2 + (ver_n - ver) ** 2)
        loss_mag = weight * ((yn_mag - y_mag) ** 2)

        # scale loss phase according the values of the pixel in the magnitude image: 
        # loss_pha = loss_pha * yn_mag

        return loss_pha + loss_mag, loss_mag, loss_pha

    def __call__(self, net, images, priors, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        imageprior = torch.cat((images,priors),dim=1) #concatenate the image and prior in the channel dimension for augmentations

        yp, augment_labels = augment_pipe(imageprior) if augment_pipe is not None else (images, None)

        y = yp[:,:images.shape[1],:,:]
        prior = yp[:,images.shape[1]:images.shape[1]*2,:,:]
        # prior_mag = yp[:,images.shape[1]*2:,:,:]
        n = torch.randn_like(y) * sigma

        ynp = torch.cat((y+n, prior),dim=1) #concatenate the image and prior in the channel dimension for input into the network as noisy y and prior
        D_yn = net(ynp, sigma, labels, augment_labels=augment_labels)

        loss, loss_mag, loss_pha = self.loss_condition(D_yn, y, weight)

        return loss, loss_mag, loss_pha

#----------------------------------------------------------------------------
