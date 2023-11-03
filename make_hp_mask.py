import numpy as np 
import pylab as plt 
import healpy as hp 
import healsparse as hsp
import pymangle
import fitsio as fio


mask_dir = "/global/cfs/cdirs/des/jelvinpo/SDSS/"

mask_files = [
    'mask_DR12v5_CMASS_North.ply',  
    'mask_DR12v5_LOWZ_North.ply',
    'mask_DR12v5_CMASS_South.ply',  
    'mask_DR12v5_LOWZ_South.ply',
    ]

nside_coverage = 32
nside_sparse = 16384
nside_output = 4096
chunk_size = 10_000_000

#for testing
#nside_coverage = 32
#nside_sparse = 512
#nside_output = 128
#chunk_size = 100_000

base_npix = hp.nside2npix(nside_sparse)
nchunks = int(np.ceil(base_npix/chunk_size))
print(f'Looping over {nchunks} chunks')

mask_file = mask_files[0]

for mask_file in mask_files:
    m = pymangle.Mangle(mask_dir + mask_file)
    
    hsp_mask = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=np.float64)
    
    #test_hp = np.zeros(base_npix)

    for ichunk in range(nchunks):
        if ichunk%100 == 0:
            print(ichunk)
        start = ichunk*chunk_size
        end = np.min( [(ichunk+1)*chunk_size, base_npix] )
        
        pix = np.arange(start,end)
        ra, dec = hp.pix2ang(nside_sparse, pix, lonlat=True, nest=True)
    
        good = m.contains(ra, dec)
    
        # get the weights
        weights = m.weight(ra,dec)
        select = good*(weights>0.0)
        #select = good
        
        hsp_mask.update_values_pix(pix[select], weights[select].astype(np.float64))
        
        #test_hp[pix[select]] = weights[select].astype(np.float64)
    
    # dont use reduction=mean, this will ignore the empty subpixels
    #hsp_mask_degraded_mean = hsp_mask.degrade(nside_output, reduction='mean')
    
    hsp_mask_degraded_sum = hsp_mask.degrade(nside_output, reduction='sum')
    select_not_empty = hsp_mask_degraded_sum[hsp_mask_degraded_sum.valid_pixels] > 0.
    hsp_mask_degraded = hsp.HealSparseMap.make_empty(nside_coverage, nside_output, dtype=np.float64)
    hsp_mask_degraded.update_values_pix(
        hsp_mask_degraded_sum.valid_pixels[select_not_empty],
        hsp_mask_degraded_sum[hsp_mask_degraded_sum.valid_pixels][select_not_empty]*(nside_output/nside_sparse)**2.
    )
    #del hsp_mask
    
    hsp_filename = mask_dir + mask_file[:-4] + f'_hsp_{nside_output}_v2.fits'
    hp_filename = mask_dir + mask_file[:-4] + f'_hp_{nside_output}_v2.fits'
    
    hsp_mask_degraded.write(hsp_filename, clobber=True)
    hp_mask = hsp_mask_degraded.generate_healpix_map(nside=nside_output)
    fio.write(hp_filename, hp_mask)
    
    plt.figure()
    hp.mollview(hp_mask, nest=True)
    plt.savefig('mask_'+mask_file[:-4]+f'_{nside_output}_v2.png')
    plt.close()

    #plt.figure()
    #hp.mollview(test_hp, nest=True)
    #plt.savefig('test_mask_'+mask_file[:-4]+'.png')
    #plt.close()

    

