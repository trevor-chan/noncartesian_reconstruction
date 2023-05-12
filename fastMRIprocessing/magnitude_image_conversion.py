import h5py
import numpy as np
import os
import glob
from PIL import Image
from multiprocessing import Pool
import sigpy



# filepath is the path to the h5 file
# outshape is the desired matrix width and height (channel dimension will always be 2 for complex)
# sampling ratio is the density of kspace sampling along each dimension - (2,1) will correspond to 
#      an image where the height of two rows equals the width of one column

def process_h5_coilcombined_magnitude(filepath,outshape=(128,128),sampling_ratio=(2,1),savedir='data_128_coilcombined'):
    if savedir is None:
        savedir = os.path.split(filepath)[0]
    savename = os.path.splitext(os.path.split(filepath)[1])[0]
    
    with h5py.File(filepath,'r') as f:
        kspace = np.array(f['kspace'])
    # recon = np.array(f['reconstruction_rss'])
        slices, coils, height, width = kspace.shape    #kspace opens as (z, coils, height, width)
        kspace_cropped = kspace[:,:,int((height/2)-(outshape[0]*sampling_ratio[0]/2)):int((height/2)+(outshape[0]*sampling_ratio[0]/2)):sampling_ratio[0], 
                                    int((width/2)-(outshape[1]*sampling_ratio[1]/2)):int((width/2)+(outshape[1]*sampling_ratio[1]/2)):sampling_ratio[1]].copy()
        
        
        for zslice in range(kspace_cropped.shape[0]):
            
            rss_array = np.sum(np.power(np.abs(sigpy.fft(kspace_cropped[zslice,:,:,:])),2),axis=0)
            savepath = '{}/{}_{}.png'.format(savedir,savename,str(zslice).zfill(2))  
            rss_array = ((rss_array * 6000000) * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(rss_array, 'L').save(savepath)
            
            

def main():
    datasetpath = os.path.expanduser("~") + '/../../' + 'mdata1/multicoil_train/*.h5'
    filelist = sorted(glob.glob(datasetpath))

    with Pool() as p:
        p.map(process_h5_coilcombined_magnitude, filelist)

if __name__=="__main__":
    main()
