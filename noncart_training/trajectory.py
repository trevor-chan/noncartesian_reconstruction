import sigpy.mri
from scipy.spatial import Voronoi, ConvexHull
from fastkde.fastKDE import pdf,pdf_at_points
import numpy as np


def generate_trajectory(matsize, interleave_range = (1,8), undersampling = 1, alpha_range = (1,4)):
    interleaves = int(np.random.rand()*(interleave_range[1]-interleave_range[0])+interleave_range[0])
    alpha = np.random.rand()*(alpha_range[1]-alpha_range[0])+alpha_range[0]

    return make_trajectory(matsize, undersampling = undersampling, interleaves = interleaves, alpha = alpha),alpha

def make_trajectory(matsize, undersampling=10, interleaves=1, alpha=1):

    fov = 1 #in meters
    adjustedshape = np.power(matsize[0]**2+matsize[1]**2,0.5)
    frequency_encode_undersampling = 1
    max_gradient_amp = 0.045 #T/m
    max_slew_rate = 0.1 #T/m/s

    if undersampling >= 0.2:
        u = .216285 / undersampling / alpha
    elif undersampling >= 0.1:
        u = .218 / undersampling / alpha
    elif undersampling >= 0.05:
        u = .23 / undersampling / alpha
    else:
        u = .26 / undersampling / alpha  #Note, at very small undersampling ratios, this approximation becomes less accurate (and probably stops being relevant anyways)
    
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
    return sigpy.interpolate(kimage, points, kernel='spline', width=2, param=1)

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
    volumes[np.isinf(volumes)] = 2
    return values*(volumes**0.25)

def simple_density_correction(points,values):   #simple correction by a factor of r universal, seems to work the best...
    values_c = values.copy()
    return values_c * np.power(np.power(points[:,0],2) + np.power(points[:,1],2), 0.5)

def fast_density_correction(points,values,alpha):   #variable correction by a factor of r^alpha universal, mixed results...
    values_c = values.copy()
    return values_c * np.power(np.power(np.power(points[:,0],2) + np.power(points[:,1],2), 0.5),0.25*alpha)

def kde_density_correction(points,values):   #extremely slow, no improvement
    values_c = values.copy()
    densities = get_pdf_at_points(points)
    return np.divide(values_c,np.abs(densities)**0.5)

def NUFFT_adjoint(points,values,matsize):
    values_c = simple_density_correction(points,values)
    image_2ch = sigpy.nufft_adjoint(values_c, points, oshape=(matsize[0], matsize[1]), oversamp=1)
    image_2ch = np.flip(np.flip(image_2ch,axis=0),axis=1)
    return image_2ch