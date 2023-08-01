#!/usr/bin/env python
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
    # parser.add_argument('-min_wavelength',default=420,type=int,help="Minimum wavelength for inversion")
    # parser.add_argument('-max_wavelength',default=950,type=int,help="Maximum wavelength for inversion")
    parser.add_argument('-apply_water_mask',default=True,type=bool,help="If True selects water by awei() > water_mask_threshold")
    parser.add_argument('-water_mask_threshold',default=0.0,type=float,help="Threshold to use for masking non-water pixels")
    parser.add_argument('-interleave',default='BIL',type=str,help="interleave")
    parser.add_argument('-dtype',default='float64',type=str,help="dtype")
    parser.add_argument('-output_format',default="ENVI",help="GDAL format to use for output raster, default ENVI")
    parser.add_argument('-glint_out',default='',type=str,help="Name for output glint map, if wanted")
    parser.add_argument('input_file',help="Image reflectance image")
    parser.add_argument('output_file',help="Output glint-corrected reflectance image")
    args = parser.parse_args()

    # Test if input_file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(f"Could not find file {args.input_file}")

    
    # Create output folder(s) for results
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir,exist_ok=True)
    if args.glint_out:
        glint_output_dir = os.path.dirname(args.glint_out)
        os.makedirs(glint_output_dir,exist_ok=True)

    ######################################################################
    ########## PREPARATION ###############################################
    ######################################################################

    # Use rioxarray to open image
    img = rioxarray.open_rasterio(args.input_file)
    # Get wavelengths information from xarray object
    wavelengths = img.wavelength.values
    # Create wavelength mask
    # wl_mask = np.logical_not(utils.band_mask(wavelengths, [[args.min_wavelength,args.max_wavelength]]))

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
        glint_profile = profile.copy()
        glint_profile['count'] = len(wavelengths)

        R_rs_profile = profile.copy()
        R_rs_profile['count'] = len(wavelengths)

        # Open output files and write row by row
        if args.glint_out:
            dst_glint = rio.open(args.glint_out, 'w', **glint_profile)
        else:
            dst_glint = None
        dst_R_rs = rio.open(args.output_file, 'w', **glint_profile)
        # Loop over rows of input file, compute mask and write to output file        
        for _, wind in src.block_windows():
            
            print(wind)

            # Read row from xarray object
            row = src.read(window=wind)
        
            if args.apply_water_mask==True:  
                # Apply water mask and convert from reflectance R [-] to above-water radiance reflectance r_rs [sr-1]
                r_rs_water = np.where(indices.awei(row, wavelengths) > args.water_mask_threshold, row, np.nan) / np.pi
            else:
                # Convert from reflectance R [-] to above-water radiance reflectance r_rs [sr-1]
                r_rs_water = row / np.pi
        
            # Apply glint correction
            glint_reflectance = glint.gao(r_rs_water, wavelengths, n2=n_res)
            R_rs = r_rs_water - glint_reflectance.astype('float32')

            # Write output rows into the respective files
            if dst_glint:
                dst_glint.write(glint_reflectance.astype('float32'), window=wind)
            dst_R_rs.write(R_rs, window=wind)
    if dst_glint:
        dst_glint.close()
    dst_R_rs.close()

    stop = timeit.default_timer()
    print('Processing time: ', stop - start, '')  

if __name__ == "__main__":
    main()
