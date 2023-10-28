import matplotlib.pyplot as plt
import PIL
import numpy as np
import torch
import training.trajectory as trajectory
from skimage.metrics import structural_similarity as ssim


def tensor_to_image(images, horv='v', normalize = False):
    if horv == 'h':
        if normalize:
            images = ((images-torch.min(images))/(torch.max(images)-torch.min(images))*255).to(torch.uint8).permute(2, 0, 3, 1).cpu().numpy()
        else:
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(2, 0, 3, 1).cpu().numpy()
        h,b,w,c = images.shape
        images = images.reshape(h,w*b,c)
        if images.shape[2]==1:
            return PIL.Image.fromarray(images[:,:,0], 'L')
        elif images.shape[2]==3:
            return PIL.Image.fromarray(images, 'RGB')
    else:
        if normalize:
            images = ((images-torch.min(images))/(torch.max(images)-torch.min(images))*255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        else:
            images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        h,b,w,c = images.shape
        images = images.reshape(h*b,w,c)
        if images.shape[2]==1:
            return PIL.Image.fromarray(images[:,:,0], 'L')
        elif images.shape[2]==3:
            return PIL.Image.fromarray(images, 'RGB')
    
def image_to_tensor(pil_img):
    tensor_img = torch.tensor(np.array(pil_img))
    if len(tensor_img.shape) == 2:
        tensor_img = ((tensor_img.to(torch.float32) - 128) / 127.5).to(device)
    else:
        assert len(tensor_img.shape) == 3
        tensor_img = ((tensor_img.to(torch.float32) - 128) / 127.5).permute(2, 0, 1).to(device)
        if tensor_img.shape[0] == 4: tensor_img = tensor_img[:3]
    return tensor_img.unsqueeze(0)

def images_to_tensor(pil_imgs):
    tensors = [image_to_tensor(img) for img in pil_imgs]
    return torch.cat(tensors, dim=0)
    
def image_array(images, columns=4, scale_h=3, scale_v=3, cmap='gray', constrain_range=False):
    height = int(images[0].size[1]/images[0].size[0])
    fig, axs = plt.subplots(int(len(images)/(columns+.01))+1, columns, figsize=(columns*scale_h,height*scale_v), frameon=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    if len(images)<=columns:
        for i,img in enumerate(images):
            if constrain_range:
                axs[i].imshow(img, cmap=cmap, vmin = 0, vmax = 255)
            else:
                axs[i].imshow(img, cmap=cmap)
            axs[i].set_axis_off()
    else:
        for i,img in enumerate(images):
            if constrain_range:
                axs[i//columns,i%columns].imshow(img, cmap=cmap, vmin = 0, vmax = 255)
            else:
                axs[i//columns,i%columns].imshow(img, cmap=cmap)
            axs[i//columns,i%columns].set_axis_off()

def coil_image_array(image, scale_h = 1.5):
    coils = []
    channels = image.shape[1]
    for ch in range(channels):
        coils.append(tensor_to_image(image[:,ch:ch+1]))
    image_array(coils, columns = channels//2, scale_h=scale_h)

def show_data_parcel(parcel):
    fig = plt.figure(layout='constrained', figsize=(15, 6))

    fig.set_facecolor('black')

    subfigs = fig.subfigures(2, 1, wspace=0.1, height_ratios=[1.4, 1])
    subfigs[0].set_facecolor('black')
    subfigs[1].set_facecolor('black')

    axs_top = subfigs[0].subplots(1, 4)
    
    im = axs_top[0].imshow(trajectory.root_summed_squares(trajectory.float_to_complex(parcel[0])).transpose(1,2,0), cmap='gray')
    axs_top[0].set_axis_off()

    axs_top[1].imshow(trajectory.root_summed_squares(trajectory.float_to_complex(parcel[1])).transpose(1,2,0), cmap='gray')
    axs_top[1].set_axis_off()

    axs_top[2].imshow(np.log(trajectory.root_summed_squares(np.array(parcel[3]['kspace'])).transpose(1,2,0)-0.5), cmap='gray')
    axs_top[2].set_axis_off()

    axs_top[3].scatter(x = parcel[3]['points'][0][:, 0], y = parcel[3]['points'][0][:, 1], s=0.1, marker='o', c='w')
    axs_top[3].axis('equal')
    axs_top[3].set_aspect('equal')
    axs_top[3].set_axis_off()
    cb = plt.colorbar(im, ax=axs_top[3], fraction=0.051)

    cb.ax.yaxis.set_tick_params(color='w')
    cb.outline.set_edgecolor('w')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='w')


    subfigsnest = subfigs[1].subfigures(2, 1, wspace=0)
    subfigsnest[0].set_facecolor('black')
    subfigsnest[1].set_facecolor('black')

    axs_bot_image = subfigsnest[0].subplots(2, parcel[0].shape[1]//2)
    axs_bot_prior = subfigsnest[1].subplots(2, parcel[0].shape[1]//2)

    image = trajectory.float_to_complex(parcel[0])
    channels = image.shape[1]
    for ch in range(channels):
        axs_bot_image[0,ch].imshow(np.abs(np.array(image[0,ch])), cmap='gray')
        axs_bot_image[0,ch].set_axis_off()
    for ch in range(channels):
        axs_bot_image[1,ch].imshow(np.angle(np.array(image[0,ch])), cmap='gray')
        axs_bot_image[1,ch].set_axis_off()

    prior = trajectory.float_to_complex(parcel[1])
    channels = prior.shape[1]
    for ch in range(channels):
        axs_bot_prior[0,ch].imshow(np.abs(np.array(prior[0,ch])), cmap='gray')
        axs_bot_prior[0,ch].set_axis_off()
    for ch in range(channels):
        axs_bot_prior[1,ch].imshow(np.angle(np.array(prior[0,ch])), cmap='gray')
        axs_bot_prior[1,ch].set_axis_off()

    plt.show()



def show_recon_parcel(parcel, recon):
    fig = plt.figure(layout='constrained', figsize=(15, 12))

    fig.set_facecolor('black')

    subfigs = fig.subfigures(2, 1, wspace=0.1, height_ratios=[1.4, 1])
    subfigs[0].set_facecolor('black')
    subfigs[1].set_facecolor('black')

    axs_top = subfigs[0].subplots(2, 4)
    
    gt_img    = trajectory.root_summed_squares(trajectory.float_to_complex(parcel[0])).transpose(1,2,0)
    prior_img = trajectory.root_summed_squares(trajectory.float_to_complex(parcel[1])).transpose(1,2,0)
    recon_img = trajectory.root_summed_squares(recon).transpose(1,2,0)
    residuals_img = np.abs((gt_img-np.amin(gt_img))/(np.amax(gt_img)-np.amin(gt_img)) - (recon_img-np.amin(recon_img))/(np.amax(recon_img)-np.amin(recon_img)))
    
    prior_ssim = ssim(gt_img[:,:,0], prior_img[:,:,0], data_range=np.amax(gt_img) - np.amin(gt_img))
    recon_ssim = ssim(gt_img[:,:,0], recon_img[:,:,0], data_range=np.amax(gt_img) - np.amin(gt_img))
    
    axs_top[0,0].imshow(gt_img, cmap='gray')
    axs_top[0,0].set_axis_off()

    axs_top[0,1].imshow(prior_img, cmap='gray')
    axs_top[0,1].set_axis_off()
    axs_top[0,1].set_title('SSIM: {:.3f}'.format(prior_ssim), color='w')
    
    axs_top[0,2].imshow(recon_img, cmap='gray')
    axs_top[0,2].set_axis_off()
    axs_top[0,2].set_title('SSIM: {:.3f}'.format(recon_ssim), color='w')
    
    im = axs_top[0,3].imshow(residuals_img, cmap='gray', vmin=0, vmax=1)
    axs_top[0,3].set_axis_off()
    
    cb = plt.colorbar(im, ax=axs_top[0,3], fraction=0.051)
    cb.ax.yaxis.set_tick_params(color='w')
    cb.outline.set_edgecolor('w')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='w')
    
    kspace_gt = trajectory.root_summed_squares(np.array(parcel[3]['kspace'])).transpose(1,2,0)-0.5
    kspace_recon = trajectory.root_summed_squares(np.array(np.fft.fftshift(np.fft.fft2(recon, axes=(-2,-1)))).astype(np.complex64)).transpose(1,2,0)-0.5
    kspace_residuals = np.abs((kspace_gt-np.amin(kspace_gt))/(np.amax(kspace_gt)-np.amin(kspace_gt)) - (kspace_recon-np.amin(kspace_recon))/(np.amax(kspace_recon)-np.amin(kspace_recon)))

    axs_top[1,0].imshow(np.log(kspace_gt), cmap='gray')
    axs_top[1,0].set_axis_off()

    axs_top[1,1].scatter(x = parcel[3]['points'][0][:, 0], y = parcel[3]['points'][0][:, 1], s=0.1, marker='o', c='w')
    axs_top[1,1].axis('equal')
    axs_top[1,1].set_aspect('equal')
    axs_top[1,1].set_axis_off()
    
    axs_top[1,2].imshow(np.log(kspace_recon), cmap='gray')
    axs_top[1,2].set_axis_off()

    im = axs_top[1,3].imshow(np.abs(kspace_residuals)*10, cmap='gray')
    axs_top[1,3].set_axis_off()
    
    cb = plt.colorbar(im, ax=axs_top[1,3], fraction=0.051)
    cb.ax.yaxis.set_tick_params(color='w')
    cb.outline.set_edgecolor('w')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='w')

    subfigsnest = subfigs[1].subfigures(3, 1, wspace=0)
    subfigsnest[0].set_facecolor('black')
    subfigsnest[1].set_facecolor('black')
    subfigsnest[2].set_facecolor('black')

    axs_bot_image = subfigsnest[0].subplots(2, parcel[0].shape[1]//2)
    axs_bot_prior = subfigsnest[1].subplots(2, parcel[0].shape[1]//2)
    axs_bot_recon = subfigsnest[2].subplots(2, parcel[0].shape[1]//2)

    image = trajectory.float_to_complex(parcel[0])
    channels = image.shape[1]
    for ch in range(channels):
        axs_bot_image[0,ch].imshow(np.abs(np.array(image[0,ch])), cmap='gray')
        axs_bot_image[0,ch].set_axis_off()
    for ch in range(channels):
        axs_bot_image[1,ch].imshow(np.angle(np.array(image[0,ch])), cmap='gray')
        axs_bot_image[1,ch].set_axis_off()

    prior = trajectory.float_to_complex(parcel[1])
    channels = prior.shape[1]
    for ch in range(channels):
        axs_bot_prior[0,ch].imshow(np.abs(np.array(prior[0,ch])), cmap='gray')
        axs_bot_prior[0,ch].set_axis_off()
    for ch in range(channels):
        axs_bot_prior[1,ch].imshow(np.angle(np.array(prior[0,ch])), cmap='gray')
        axs_bot_prior[1,ch].set_axis_off()

    for ch in range(channels):
        axs_bot_recon[0,ch].imshow(np.abs(np.array(recon[0,ch])), cmap='gray')
        axs_bot_recon[0,ch].set_axis_off()
    for ch in range(channels):
        axs_bot_recon[1,ch].imshow(np.angle(np.array(recon[0,ch])), cmap='gray')
        axs_bot_recon[1,ch].set_axis_off()

    plt.show()
