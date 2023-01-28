import sigpy.mri
from scipy.spatial import Voronoi, ConvexHull
from fastkde.fastKDE import pdf,pdf_at_points
import numpy as np


def generate_trajectory(matsize, interleave_range = (1,8), undersampling = 1, alpha_range = (1,4)):
    interleaves = int(np.random.rand()*(interleave_range[1]-interleave_range[0])+interleave_range[0])
    alpha = np.random.rand()*(alpha_range[1]-alpha_range[0])+alpha_range[0]

    return make_trajectory(matsize, undersampling = undersampling, interleaves = interleaves, alpha = alpha),alpha

def make_trajectory(matsize, undersampling=1, interleaves=1, alpha=1):

    fov = .25 #in meters
    adjustedshape = np.power(matsize[0]**2+matsize[1]**2,0.5)
    frequency_encode_undersampling = 1
    max_gradient_amp = 0.045 #T/m  4.5 G/cm, or 2.2 G/cm
    max_slew_rate = 0.1 #T/m/ms

    u = .0365 / undersampling * (1-2*(((1/alpha)**4)-(1/alpha)**2)) # Empirically determined as a decent approximation for maintaining a steady effective undersampling rate
    
    interleaves = interleaves

    points = sigpy.mri.spiral(fov, 
                            adjustedshape, 
                            frequency_encode_undersampling, 
                            u, 
                            interleaves, 
                            alpha, 
                            max_gradient_amp, 
                            max_slew_rate)

    points = np.asarray([i for i in points if -matsize[0]/2<=i[0]<=matsize[0]/2 and -matsize[1]/2<=i[1]<=matsize[1]/2]) #trim excess
    information_ratio = points.shape[0] / (matsize[0]*matsize[1]) #recalculate the new information sampling

    # print('information_ratio: {}'.format(information_ratio))

    return points

def interpolate_values(points,kimage):
    assert kimage.dtype == 'complex64'
    # return sigpy.interpolate(kimage, points, kernel='spline', width=2, param=1)
    mag_image = np.abs(sigpy.ifft(kimage))
    return sigpy.nufft(mag_image, points)

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol

def get_pdf_at_points(points):
    test_points = list(points)
    return pdf_at_points(points[:,0], points[:,1], list_of_points = test_points)

def voronoi_density_correction(points,values): #moderately slow, no improvement
    volumes = voronoi_volumes(points)
    volumes[np.isinf(volumes)] = 1
    return values*(volumes**0.5)

def simple_density_correction(points,values):   #simple correction by a factor of r universal
    values_c = values.copy()
    return values_c * np.power(np.power(points[:,0],2) + np.power(points[:,1],2), 0.5)

def fast_density_correction(points,values,alpha,scalefactor=0.5): #Adaptive correction by a factor of r^(1-5^(1-alpha)/2), very fast to compute
    values_c = values.copy()
    return values_c * np.power(np.power(points[:,0],2) + np.power(points[:,1],2), (1-5**(1-alpha))/2 ) #Likewise empirically determined to give the best quality results, though subject to change



def kde_density_correction(points,values):   #extremely slow, no improvement
    values_c = values.copy()
    densities = get_pdf_at_points(points)
    return np.divide(values_c,np.abs(densities))

def NUFFT_adjoint(points,values,matshape,alpha):
    values_c = fast_density_correction(points,values,alpha)
    image_2ch = sigpy.nufft_adjoint(values_c, points, oshape=matshape, oversamp=1)
    image_2ch = np.flip(np.flip(image_2ch,axis=0),axis=1)
    return image_2ch