#!/usr/bin/env python
import numpy as np
import rasterio as rio
import rioxarray
import argparse
import os
import warnings
import timeit
from datetime import datetime

from bio_optics.helper import utils, indices, resampling
from bio_optics.models import model, spm
from bio_optics import surface
from bio_optics.surface import glint
from bio_optics.atmosphere import downwelling_irradiance

import json

##Ignore some warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)
np.seterr(invalid='ignore')

######################################################################
########## MAIN ######################################################
######################################################################

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Invert Reflectance [-] image')
    parser.add_argument('--output_glint',action='store_true',help="Write glint spectrum to file")
    parser.add_argument('--output_score',action='store_true',help="Write glint score to file")
    parser.add_argument('--output_rrs',action='store_true',help="Write Rrs spectrum to file")
    parser.add_argument('--min_wavelength',default=420,type=int,help="Minimum wavelength for inversion")
    parser.add_argument('--max_wavelength',default=950,type=int,help="Maximum wavelength for inversion")
    parser.add_argument('--interleave',default='BIL',type=str,help="interleave")
    parser.add_argument('--dtype',default='float64',type=str,help="dtype")
    parser.add_argument('--output_format',default="ENVI",help="GDAL format to use for output raster, default ENVI")
    parser.add_argument('input_file',help="Reflectance Image file to invert for")
    args = parser.parse_args()

    # Test if input_file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(f"Could not find file {args.input_file}")

    
    # Create output folder for results
    #  output_dir = ('_').join([os.path.abspath(args.input_file), 'spm', datetime.now().strftime('%Y%m%d%H%M%S')])
    #  if not os.path.exists(output_dir):
    #      # If it doesn't exist, create it
    #      os.makedirs(output_dir)

    ######################################################################
    ########## PREPARATION ###############################################
    ######################################################################

    # Use rioxarray to open image
    print(f"Opening input raster {args.input_file}")
    img = rioxarray.open_rasterio(args.input_file)
    # Get wavelengths information from xarray object
    wavelengths = img.wavelength.values
    # Create wavelength mask
    wl_mask = np.logical_not(utils.band_mask(wavelengths, [[args.min_wavelength,args.max_wavelength]]))
    score_mask = np.logical_not(utils.band_mask(wavelengths, [[500,560]]))
    print(f"Found {wl_mask.sum()} bands in wl range {[args.min_wavelength,args.max_wavelength]}")

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
        profile['nodata'] = "-9999"
        if "transform" in profile:
            profile.pop("transform")

        # prepare individual profiles for each of the outputs
        spm_profile = profile.copy()
        spm_profile['count'] = 1

        glint_profile = profile.copy()
        glint_profile['count'] = len(wavelengths)

        score_profile = profile.copy()
        score_profile['count'] = 1

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
        def make_ouput_name(inpath, suffix):
            parts = os.path.splitext(inpath)
            return suffix.join(parts)


        if args.output_glint:
            outglint = make_ouput_name(args.input_file, "_gao_glint")
            print(f"Creating output {outglint}")
            dst_glint = rio.open(outglint, 'w', **glint_profile)
        if args.output_rrs:
            outrrs = make_ouput_name(args.input_file, "_rrs")
            print(f"Creating output {outrrs}")
            dst_rrs = rio.open(outrrs, 'w', **rrs_profile)
        if args.output_score:
            outscore = make_ouput_name(args.input_file, "_glint_score.tif")
            print(f"Creating output {outscore}")
            dst_score = rio.open(outscore, 'w', **score_profile)

        outspm = make_ouput_name(args.input_file, "_spm")
        print(f"Creating output {outspm}")

        with rio.open(outspm, 'w', **spm_profile) as dst_spm:
               
                # Loop over rows of input file, compute mask and write to output file        
                reports = (np.arange(21)*5).tolist()
                blocks = [b[1] for b in list(src.block_windows())]
                numblocks = len(blocks)
                blockshape = (blocks[0].width, blocks[0].height)
                print(f"Processing {numblocks} blocks of shape {blockshape}")
                for w_i, wind in enumerate(blocks):
                    perc = ((w_i + 1) / numblocks * 100)
                    if reports and (perc > reports[0]):
                        report_perc = reports.pop(0)
                        print(f"{report_perc:>3d}%")
                    
                    # Read row from xarray object
                    row = src.read(window=wind)
                
                    # Apply water mask and convert from reflectance R [-] to above-water radiance reflectance r_rs [sr-1]
                    r_rs_water = np.where(indices.awei(row, wavelengths) > 0, row, np.nan) / np.pi
                
                    # Apply glint correction
                    glint_reflectance = glint.gao(r_rs_water, wavelengths, n2=n_res)
                    if args.output_score:
                        ##output avg of bands for wavelengths 500-560
                        glint_score = glint_reflectance[score_mask].mean(axis=0)
                    R_rs = r_rs_water - glint_reflectance.astype('float32')

                    # Compute spm
                    out_spm = spm.jiang(R_rs[wl_mask], wavelengths[wl_mask])

                    # Write output rows into the respective files
                    def format_arr(inarr):
                        return np.where(np.isfinite(inarr),inarr,-9999).astype('float32')
                    dst_spm.write(format_arr(out_spm[np.newaxis,:]), window=wind)
                    if args.output_glint:
                        dst_glint.write(format_arr(glint_reflectance), window=wind)
                    if args.output_score:
                        dst_score.write(format_arr(glint_score[np.newaxis,:]), window=wind)
                    if args.output_rrs:
                        dst_R_rs.write(format_arr(R_rs), window=wind)

    ##Cleanup
    if args.output_glint:
        dst_glint.close()
        dst_glint = None
    if args.output_score:
        dst_score.close()
        dst_score = None
    if args.output_rrs:
        dst_R_rs.close()
        dst_R_rrs = None
    stop = timeit.default_timer()
    print('Processing time: ', stop - start, '')  

if __name__ == "__main__":
    main()
