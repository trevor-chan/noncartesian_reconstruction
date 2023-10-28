import sigpy.mri
import pynufft
from scipy.spatial import Voronoi, ConvexHull
from fastkde.fastKDE import pdf,pdf_at_points
import numpy as np
import seaborn as sns
import torch
import PIL


def generate_trajectory(matsize, interleave_range = (1,8), undersampling = 1, alpha_range = (1,4)):
    interleaves = int(np.random.rand()*(interleave_range[1]-interleave_range[0])+interleave_range[0])
    alpha = np.random.rand()*(alpha_range[1]-alpha_range[0])+alpha_range[0]

    return make_trajectory(matsize, undersampling = undersampling, interleaves = interleaves, alpha = alpha)

def make_trajectory(matsize, undersampling=1, interleaves=1, alpha=1):

    fov = .22 #in meters
    # adjustedshape = np.power(matsize[0]**2+matsize[1]**2,0.5)
    adjustedshape = matsize[0]
    frequency_encode_undersampling = 1 # default to 1
    max_gradient_amp = 0.045 #T/m  4.5 G/cm, or 2.2 G/cm       - Brian uses 0.040
    max_slew_rate = 0.2 * 1000 #T/m/s

    # u = .0365 / undersampling * (1-2*(((1/alpha)**4)-(1/alpha)**2)) # Empirically determined as a decent approximation for maintaining a steady effective undersampling rate
    u = undersampling
    
    points = sigpy.mri.spiral(fov, 
                            adjustedshape, 
                            frequency_encode_undersampling, 
                            u, 
                            interleaves, 
                            alpha, 
                            max_gradient_amp, 
                            max_slew_rate)

    # points = np.asarray([i for i in points if -matsize[0]/2<=i[0]<=matsize[0]/2 and -matsize[1]/2<=i[1]<=matsize[1]/2]) # square 

    # points = np.delete(points, np.where((np.hypot(points[:,0],points[:,1]) >= matsize[0]/2)), axis=0) # circle
    
    # information_ratio = points.shape[0] / (matsize[0]*matsize[1]) #recalculate the new information sampling
    # print('information_ratio: {}'.format(information_ratio))

    return points

# perform an interpolation in kspace
# if given a single image and an array of points, returns an array of values. If given a batch and a list of arrays of points returns a list of arrays of values
def interpolate_values(points, kspace, width=2):
    assert isinstance(kspace,np.ndarray), f'expected kimage to be a numpy ndarray but got type {type(kspace)}'
    assert kspace.dtype == np.complex64, f'expected kimage to have a datatype of np.complex64 but got type {kspace.dtype}'
    assert len(kspace.shape) > 2 & len(kspace.shape) < 5, f'expected kimage shape to be in form [ch, w, h] or [b, ch, w, h], got {kspace.shape}'

    kspace = np.fft.fftshift(kspace, axes = (-1,-2))
    if len(kspace.shape) == 3:
        assert isinstance(points,np.ndarray), f'expected points to a numpy ndarray but got type {type(points)}'
        assert len(points.shape) == 2, f'expected points shape to be in form [samples, xydim], got {points.shape}'

        values = np.stack([sigpy.interpolate(kspace[coil], points, kernel='kaiser_bessel', width=width) for coil in range(kspace.shape[0])], axis=0)
        return values
    else:
        assert isinstance(points,list), f'expected points to a list of numpy ndarrays but got type {type(points)}'
        values = []
        for b in range(kimage.shape[0]):
            assert len(points[b].shape) == 2, f'expected points shape to be in form [samples, xydim], got {points[b].shape}'
            values.append(np.stack([sigpy.interpolate(kspace[b,coil], points[b], kernel='kaiser_bessel', width=width) for coil in range(kspace[b].shape[0])], axis=0))
        return values

# preallocate memory for the nufft object and return, works on individual images and returns a list in the case of batch
# given a single array, returns the nufftobj; given a batch of arrays, returns a list of nufftobjs
def prealloc_nufft(image_array, points):
    assert isinstance(image_array,np.ndarray), f'expected a numpy ndarray but got type {type(image_array)}'
    assert image_array.dtype == np.dtype(np.complex64), f'expected a datatype of np.complex64 but got type {image_array.dtype}'
    assert len(image_array.shape) > 2 & len(image_array.shape) < 4, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {image_array.shape}'

    if len(image_array.shape) == 3:
        assert isinstance(points,np.ndarray), f'expected points to a numpy ndarray but got type {type(points)}'
        nufftobj = pynufft.NUFFT()

        om = points
        om = om/np.amax(om)*np.pi
        # om = np.delete(om, np.where((np.hypot(om[:,0],om[:,1]) >= np.pi)), axis=0)

        Nd = (image_array.shape[-2],image_array.shape[-1])
        Kd = (Nd[0]*2,Nd[1]*2)
        Jd = (Nd[0]//32,Nd[1]//32)

        nufftobj.plan(om, Nd, Kd, Jd)

        return nufftobj
    else:
        # assert isinstance(points,list), f'expected points to a list of numpy ndarrays but got type {type(points)}'
        nufftobjs = []
        for i in range(image_array.shape[0]):
            nufftobj = pynufft.NUFFT()
            
            om = points[i]
            om = om/np.amax(om)*np.pi
            # om = np.delete(om, np.where((np.hypot(om[:,0],om[:,1]) >= np.pi)), axis=0)

            Nd = (image_array.shape[-2],image_array.shape[-1])
            Kd = (Nd[0]*2,Nd[1]*2)
            Jd = (Nd[0]//32,Nd[1]//32)

            nufftobj.plan(om, Nd, Kd, Jd)

            nufftobjs.append(nufftobj)

        return nufftobjs
    
# compute forward nufft
def forward_nufft(x, nufftobjs):
    assert isinstance(x,np.ndarray), f'expected a numpy ndarray but got type {type(x)}'
    assert x.dtype == np.dtype(np.complex64), f'expected a datatype of np.complex64 but got type {x.dtype}'
    assert len(x.shape) > 2 & len(x.shape) < 5, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {x.shape}'

    if len(x.shape) == 3:
        y = np.zeros((x.shape[0], nufftobjs.M[0]), dtype=np.complex64)
        channels = np.count_nonzero(np.count_nonzero(np.sum(x,axis=-1), axis=-1))
        for ch in range(channels):
            y[ch] = nufftobjs.forward(x[ch])
    
    else:
        assert isinstance(nufftobjs,list), 'provided a batch of images, requires a list of preallocated nufft objects'
        batches = x.shape[0]
        for b in range(batches):
            y = np.zeros((x.shape[0], x.shape[1], nufftobjs[b].M[0]), dtype=np.complex64)
            channels = np.count_nonzero(np.count_nonzero(np.sum(x[b],axis=-1), axis=-1))
            for ch in range(channels):
                y[b,ch] = nufftobjs[b].forward(x[b,ch])
    return y

# compute inverse nufft
def inverse_nufft(y, nufftobjs, maxiter=100):
    assert isinstance(y,np.ndarray), f'expected a numpy ndarray but got type {type(y)}'
    assert y.dtype == np.dtype(np.complex64), f'expected a datatype of np.complex64 but got type {y.dtype}'
    assert len(y.shape) > 1 & len(y.shape) < 4, f'expected shape to be in form [ch, sample] or [b, ch, sample], got {y.shape}'

    if len(y.shape) == 2:
        x = np.zeros((y.shape[0],nufftobjs.Nd[0],nufftobjs.Nd[1]), dtype=np.complex64)
        channels = np.count_nonzero(np.count_nonzero(y, axis=-1))
        for ch in range(channels):
            x[ch] = nufftobjs.solve(y[ch], solver='cg',maxiter=maxiter)
    
    else:
        assert isinstance(nufftobjs,list), 'provided a batch of frequency samples, requires a list of preallocated nufft objects'
        batches = y.shape[0]
        for b in range(batches):
            x = np.zeros((y.shape[0],y.shape[1],nufftobjs[b].Nd[0],nufftobjs[b].Nd[1]), dtype=np.complex64)
            channels = np.count_nonzero(np.count_nonzero(y[b], axis=-1))
            for ch in range(channels):
                 x[b,ch] = nufftobjs[b].solve(y[b,ch], solver='cg',maxiter=maxiter)
    return x


# normalize the intensity of a prior image by setting the standard deviation
def intensity_normalization(image_array, deviations=4, clipping=True):
    assert isinstance(image_array,np.ndarray), f'expected a numpy ndarray but got type {type(image_array)}'
    assert image_array.dtype == np.dtype(np.complex64), f'expected a datatype of np.complex64 but got type {image_array.dtype}'
    assert len(image_array.shape) > 2 & len(image_array.shape) < 5, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {image_array.shape}'

    def sym_clip(arr,lim): 
        np.clip(arr.real, -lim, lim, out=arr.real)
        np.clip(arr.imag, -lim, lim, out=arr.imag)

    if len(image_array.shape) == 3:
        channels = np.nonzero(np.sum(image_array,axis=(-1,-2)))[0]
        stds = []
        for ch in channels:
            stds.append(np.std(image_array[ch]))
        std_mean = sum(stds)/len(stds)
        image_array = image_array * (1/std_mean) / deviations #scale data such that +-1 lies at 3 standard devations from the mean
        if clipping: sym_clip(image_array, 1)
        return image_array
    else:
        for b in range(image_array.shape[0]):
            channels = np.nonzero(np.sum(image_array[b],axis=(-1,-2)))[0]
            stds = []
            for ch in channels:
                stds.append(np.std(image_array[b,ch]))
            std_mean = sum(stds)/len(stds)
            image_array[b] = image_array[b] * (1/std_mean) / deviations
        if clipping: sym_clip(image_array, 1)
        return image_array


def plot_trajectory(points, height=5, ax = None):
    sns.set_palette('CMRmap')
    sns.jointplot(x=points[:,0],y=points[:,1],marginal_kws=dict(bins=50), marker=".", s=5, height=height)
    # sns.jointplot(points,height=3,marker=".")

# Perform RSS reconstruction on a pytorch tensor of shape batch, Channels/magphase(interspersed), H, W
def root_summed_squares(image_array, phase = False):
    assert isinstance(image_array,np.ndarray), f'expected a numpy ndarray but got type {type(image_array)}'
    assert image_array.dtype == np.dtype(np.complex64), f'expected a datatype of np.complex64 but got type {image_array.dtype}'
    assert len(image_array.shape) > 2 & len(image_array.shape) < 5, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {image_array.shape}'

    if phase:
        phase_array = np.angle(image_array) / np.pi
        channels = image_array.shape[-3]
        phase_combined = np.power(np.sum(np.power(phase_array,2)/channels,axis=-3),0.5)
        return phase_combined
    else:
        magnitude_array = (np.abs(image_array)+1)/2
        channels = image_array.shape[-3]
        magnitude_combined = np.power(np.sum(np.power(magnitude_array,2)/channels,axis=-3),0.5)
        return magnitude_combined

# convert numpy array of type complex64 and shape [ch, w, h] or [b, ch, w, h] to torch tensor of type float32 and shape [2*ch, w, h] or [b, 2*ch, w, h]
def complex_to_float(image_array, device = None):
    assert isinstance(image_array,np.ndarray) or isinstance(image_array,torch.Tensor) , f'expected a numpy ndarray or torch tensor but got type {type(image_array)}'
    assert image_array.dtype == np.dtype(np.complex64) or image_array.dtype == torch.complex64, f'expected a datatype of np.complex64 but got type {image_array.dtype}'
    assert len(image_array.shape) > 2 & len(image_array.shape) < 5, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {image_array.shape}'

    real = image_array.real
    imag = image_array.imag

    if isinstance(image_array,np.ndarray):
        temp_array = np.concatenate((real, imag), axis=-3)
        return torch.tensor(temp_array, dtype=torch.float32, device=device)
    else:
        temp_array = torch.cat((real, imag), dim=-3)
        return temp_array.to(torch.float32).to(device)


# convert torch tensor of type float32 and shape [2*ch, w, h] or [b, 2*ch, w, h] to numpy array of type complex64 and shape [ch, w, h] or [b, ch, w, h]
def float_to_complex(image_tensor, device = None):
    assert isinstance(image_tensor,torch.Tensor), f'expected a torch tensor but got type {type(image_tensor)}'
    assert image_tensor.dtype == torch.float32, f'expected a datatype of torch.float32 but got type {image_tensor.dtype}'
    assert len(image_tensor.shape) > 2 & len(image_tensor.shape) < 5, f'expected shape to be in form [2*ch, w, h] or [b, 2*ch, w, h], got {image_tensor.shape}'
    assert image_tensor.shape[-3] % 2 == 0, 'channel dimension is not even, what are you doing?'

    image_tensor = np.array(image_tensor.cpu())

    if len(image_tensor.shape) == 3:
        real = image_tensor[:image_tensor.shape[-3]//2]
        imag = image_tensor[image_tensor.shape[-3]//2:] * 1j
    else:
        real = image_tensor[:, :image_tensor.shape[-3]//2]
        imag = image_tensor[:, image_tensor.shape[-3]//2:] * 1j

    return (real+imag).astype(np.complex64)


# convert numpy array of type complex64 and shape [ch, w, h] or [b, ch, w, h] to torch tensor of type float32 and shape [2*ch, w, h] or [b, 2*ch, w, h]
def complex_to_magphase(image_array, device = None):
    assert isinstance(image_array,np.ndarray) or isinstance(image_array,torch.Tensor) , f'expected a numpy ndarray or torch tensor but got type {type(image_array)}'
    assert image_array.dtype == np.dtype(np.complex64) or image_array.dtype == torch.complex64, f'expected a datatype of np.complex64 but got type {image_array.dtype}'
    assert len(image_array.shape) > 2 & len(image_array.shape) < 5, f'expected shape to be in form [ch, w, h] or [b, ch, w, h], got {image_array.shape}'

    if isinstance(image_array,np.ndarray):
        mag = np.abs(image_array)
        pha = np.angle(image_array) / np.pi
        temp_array = np.concatenate((mag, pha), axis=-3)
        return torch.tensor(temp_array, dtype=torch.float32, device=device)
    else:
        mag = torch.abs(image_array)
        pha = torch.angle(image_array) / torch.pi
        temp_array = torch.cat((mag, pha), dim=-3)
        return temp_array.to(torch.float32).to(device)
    

# convert torch tensor of type float32 and shape [2*ch, w, h] or [b, 2*ch, w, h] to numpy array of type complex64 and shape [ch, w, h] or [b, ch, w, h]
def magphase_to_complex(image_tensor, device = None):
    assert isinstance(image_tensor,torch.Tensor), f'expected a torch tensor but got type {type(image_tensor)}'
    # assert image_tensor.dtype == torch.float32, f'expected a datatype of torch.float32 but got type {image_tensor.dtype}'
    assert len(image_tensor.shape) > 2 & len(image_tensor.shape) < 5, f'expected shape to be in form [2*ch, w, h] or [b, 2*ch, w, h], got {image_tensor.shape}'
    assert image_tensor.shape[-3] % 2 == 0, 'channel dimension is not even, what are you doing?'

    if len(image_tensor.shape) == 3:
        complex = image_tensor[:image_tensor.shape[1]//2] * (torch.cos(image_tensor[image_tensor.shape[1]//2:]*torch.pi) + torch.sin(image_tensor[image_tensor.shape[1]//2:]*torch.pi) * 1j)
    else:
        complex = image_tensor[:,:image_tensor.shape[1]//2] * (torch.cos(image_tensor[:,image_tensor.shape[1]//2:]*torch.pi) + torch.sin(image_tensor[:,image_tensor.shape[1]//2:]*torch.pi) * 1j)

    complex = np.array(complex.cpu())
    return complex.astype(np.complex64)

