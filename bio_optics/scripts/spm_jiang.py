import numpy as np
import rasterio as rio
import rioxarray
import argparse
import os
import timeit
from datetime import datetime

from bio_optics.helper import utils, indices, resampling
from bio_optics.models import model, spm
from bio_optics import surface
from bio_optics.surface import glint
from bio_optics.atmosphere import downwelling_irradiance

import json


######################################################################
########## MAIN ######################################################
######################################################################

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Invert Reflectance [-] image')
    parser.add_argument('-input_file',help="Image file to invert for")
    parser.add_argument('-min_wavelength',default=420,type=int,help="Minimum wavelength for inversion")
    parser.add_argument('-max_wavelength',default=950,type=int,help="Maximum wavelength for inversion")
    parser.add_argument('-interleave',default='BIL',type=str,help="interleave")
    parser.add_argument('-dtype',default='float64',type=str,help="dtype")
    parser.add_argument('-output_format',default="ENVI",help="GDAL format to use for output raster, default ENVI")
    args = parser.parse_args()

    # Test if input_file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(f"Could not find file {args.input_file}")

    
    # Create output folder for results
    output_dir = ('_').join([os.path.abspath(args.input_file), 'spm', datetime.now().strftime('%Y%m%d%H%M%S')])
    if not os.path.exists(output_dir):
        # If it doesn't exist, create it
        os.makedirs(output_dir)

    ######################################################################
    ########## PREPARATION ###############################################
    ######################################################################

    # Use rioxarray to open image
    img = rioxarray.open_rasterio(args.input_file)
    # Get wavelengths information from xarray object
    wavelengths = img.wavelength.values
    # Create wavelength mask
    wl_mask = np.logical_not(utils.band_mask(wavelengths, [[args.min_wavelength,args.max_wavelength]]))

    # Resample LUTs
    n_res = resampling.resample_n(wavelengths)

    ######################################################################
    ########## LOOP OVER ROWS ############################################
    ######################################################################

    start = timeit.default_timer()

    with rio.open(args.input_file, 'r') as src:

        # Get profile
        profile = src.profile
        profile['driver'] = args.output_format
        profile['interleave'] = args.interleave
        profile['dtype'] = args.dtype

        # prepare individual profiles for each of the outputs
        spm_profile = profile.copy()
        spm_profile['count'] = 1

        glint_profile = profile.copy()
        glint_profile['count'] = len(wavelengths)

        R_rs_profile = profile.copy()
        R_rs_profile['count'] = len(wavelengths)

        # # Create output files and close right away - this was a try to avoid the killings on Agave
        # dst_spm = rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_spm']), 'w', **spm_profile)
        # dst_spm.close()

        # dst_glint = rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_glint']), 'w', **glint_profile)
        # dst_glint.close()

        # dst_R_rs = rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_Rrs']), 'w', **R_rs_profile)
        # dst_R_rs.close()

        # Open output files and write row by row
        with rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_spm']), 'w', **spm_profile) as dst_spm, \
             rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_glint']), 'w', **glint_profile) as dst_glint, \
             rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_Rrs']), 'w', **R_rs_profile) as dst_R_rs:
               
                # Loop over rows of input file, compute mask and write to output file        
                for _, wind in src.block_windows():
                    
                    print(wind)

                    # Read row from xarray object
                    row = src.read(window=wind)
                
                    # Apply water mask and convert from reflectance R [-] to above-water radiance reflectance r_rs [sr-1]
                    r_rs_water = np.where(indices.awei(row, wavelengths) > 0, row, np.nan) / np.pi
                
                    # Apply glint correction
                    glint_reflectance = glint.gao(r_rs_water, wavelengths, n2=n_res)
                    R_rs = r_rs_water - glint_reflectance.astype('float32')

                    # Compute spm
                    out_spm = spm.jiang(R_rs[wl_mask], wavelengths[wl_mask])

                    # Write output rows into the respective files
                    dst_spm.write(out_spm[np.newaxis,:].astype('float32'), window=wind)
                    dst_glint.write(glint_reflectance.astype('float32'), window=wind)
                    dst_R_rs.write(R_rs, window=wind)

    stop = timeit.default_timer()
    print('Processing time: ', stop - start, '')  

if __name__ == "__main__":
    main()