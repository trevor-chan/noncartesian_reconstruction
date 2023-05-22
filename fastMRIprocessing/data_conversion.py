import h5py
import numpy as np
import os
import glob
from PIL import Image
from multiprocessing import Pool



# filepath is the path to the h5 file
# outshape is the desired matrix width and height (channel dimension will always be 2 for complex)
# sampling ratio is the density of kspace sampling along each dimension - (2,1) will correspond to 
#      an image where the height of two rows equals the width of one column

def process_h5_singlecoil(filepath,outshape=(256,256),sampling_ratio=(2,1),savedir='data_256'):
    print(filepath)
    if savedir is None:
        savedir = os.path.split(filepath)[0]
    savename = os.path.splitext(os.path.split(filepath)[1])[0]
    
    with h5py.File(filepath,'r') as f:
        kspace = np.array(f['kspace'])
        # recon = np.array(f['reconstruction_rss'])
        if kspace.shape[2] < outshape[0] or kspace.shape[3] < outshape[1]:
            return 0
        # assert kspace.shape[2] > outshape[0], f'kspace shape {kspace.shape}'
        # assert kspace.shape[3] > outshape[1], f'kspace shape {kspace.shape}'
        slices, coils, height, width = kspace.shape    #kspace opens as (z, coils, height, width)
        kspace_cropped = kspace[:,:,int((height/2)-(outshape[0]*sampling_ratio[0]/2)):int((height/2)+(outshape[0]*sampling_ratio[0]/2)):sampling_ratio[0], 
                                    int((width/2)-(outshape[1]*sampling_ratio[1]/2)):int((width/2)+(outshape[1]*sampling_ratio[1]/2)):sampling_ratio[1]].copy()
    
        for zslice in range(kspace_cropped.shape[0]):
            for coil in range(kspace_cropped.shape[1]):
                array = np.stack((kspace_cropped[zslice,coil,:,:].real,
                                  kspace_cropped[zslice,coil,:,:].imag), axis=-1)  #Saves in image channel format, (height, width, channel) to be permuted later
                savepath = '{}/{}_{}_{}.npy'.format(savedir,savename,str(zslice).zfill(2),str(coil).zfill(2))
                
                with open(savepath, 'wb') as f:
                    np.save(f, array)
            
            

def main():
    datasetpath = os.path.expanduser("~") + '/../../' + 'mdata1/multicoil_train/*.h5'
    filelist = sorted(glob.glob(datasetpath))

    with Pool() as p:
        p.map(process_h5_singlecoil, filelist)
        
if __name__=="__main__":
    main()
