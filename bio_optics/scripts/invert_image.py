#!/usr/bin/env python
import numpy as np
import rasterio as rio
import rioxarray
import argparse
import os
import shutil
import ray
import timeit
from datetime import datetime

from bio_optics.helper import utils, indices, resampling
from bio_optics.models import model
from bio_optics import surface
from bio_optics.atmosphere import downwelling_irradiance

import json
import lmfit

######################################################################
########## FUNCTIONS #################################################
######################################################################

def read_params(filepath):
    """
    Reads the params.json file and translates it into an lmfit.Parameters() object.

    Args:
        filepath: path to the params.json file

    Returns:
        params: lmfit.Parameters() object
    """
    # Read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Create an lmfit.Parameters object
    params = lmfit.Parameters()

    # Loop through each parameter in the JSON data and add it to the parameters object
    for param_data in data['params']:
        name = param_data['name']
        value = param_data['value']
        vary = param_data['vary']
        
        # Check if the min and max keys are present in the parameter data
        if 'min' in param_data and 'max' in param_data:
            min_val = param_data['min']
            max_val = param_data['max']
            
            # Add the parameter with min, max, and vary information
            params.add(name, value=value, min=min_val, max=max_val, vary=vary)
        else:
            # Add the parameter without min, max, and vary information
            params.add(name, value=value, vary=vary)

    return params


@ray.remote
def invert_chunk(chunk,
                 params,
                 wavelengths,
                 weights,
                 a_i_spec_res,
                 a_w_res,
                 b_phy_norm_res,
                 bb_w_res,
                 R_i_b_res,
                 E0_res,
                 a_oz_res,
                 a_ox_res,
                 a_wv_res,
                 da_w_div_dT_res,
                 Ed_res,
                 Ed_sa_res,
                 Ed_sr_res,
                 Ed_d_res,
                 method="least_squares", 
                 max_nfev=400):
    """
    Loop over a chunk of spectra and apply the inversion model.
    Ray-decorated to run in parallel.
    """
    # Create empty array to fill with results
    results = np.array([None] * chunk.shape[0])

    # Looper over elements in chunk and fill results
    for i in np.arange(chunk.shape[0]):
        inv = model.invert(params, 
                           Rrs=chunk[i].values.astype(float), 
                           wavelengths=wavelengths, 
                           weights=weights,
                           a_i_spec_res=a_i_spec_res,
                           a_w_res=a_w_res,
                           b_phy_norm_res=b_phy_norm_res,
                           bb_w_res=bb_w_res,
                           R_i_b_res=R_i_b_res,
                           E0_res = E0_res,
                           a_oz_res = a_oz_res,
                           a_ox_res = a_ox_res,
                           a_wv_res = a_wv_res,
                           da_w_div_dT_res = da_w_div_dT_res,
                           Ed_res = Ed_res,
                           Ed_sa_res = Ed_sa_res,
                           Ed_sr_res = Ed_sr_res,
                           Ed_d_res = Ed_d_res,
                           method=method, 
                           max_nfev=max_nfev)
        results[i] = inv
        
    return results


######################################################################
########## MAIN ######################################################
######################################################################

def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(description='Invert Reflectance [-] image')
    parser.add_argument('-input_file',help="Image file to invert for")
    parser.add_argument('-params_file',help="JSON file containing the fit parameters")
    parser.add_argument('-min_wavelength',default=420,type=int,help="Minimum wavelength for inversion, default: 420")
    parser.add_argument('-max_wavelength',default=850,type=int,help="Maximum wavelength for inversion, default: 850")
    parser.add_argument('-method',default='least_squares',type=str,help="Minimization method, default: least_squares")
    parser.add_argument('-max_nfev',default=1500,type=int,help="Maximum number of iterations per pixel, default: 1500")
    parser.add_argument('-interleave',default='BIL',type=str,help="interleave, default: BIL")
    parser.add_argument('-dtype',default='float64',type=str,help="dtype: default: float64")
    parser.add_argument('-output_format',default="ENVI",help="GDAL format to use for output raster, default: ENVI")
    parser.add_argument('-shade',default=False,help='Shade flag for AWEI, default: False')
    args = parser.parse_args()

    # Test if input_file exists
    if not os.path.exists(args.input_file):
        raise RuntimeError(f"Could not find file {args.input_file}")
    
    # Test if params.json exists
    if not os.path.exists(args.params_file):
        raise RuntimeError(f"Could not find file {args.params_file}")
    
    # Create output folder for results
    output_dir = ('_').join([os.path.abspath(args.input_file), 'inversion', datetime.now().strftime('%Y%m%d%H%M%S')])
    if not os.path.exists(output_dir):
        # If it doesn't exist, create it
        os.makedirs(output_dir)

    # Copy params file to output folder
    shutil.copyfile(os.path.abspath(args.params_file), ('/').join([output_dir, 'params.json']))

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
    a_i_spec_res = resampling.resample_a_i_spec(wavelengths[wl_mask])
    a_w_res = resampling.resample_a_w(wavelengths[wl_mask])
    b_phy_norm_res = resampling.resample_b_phy_norm(wavelengths[wl_mask])
    bb_w_res = resampling.resample_bb_w(wavelengths[wl_mask])
    R_i_b_res = resampling.resample_R_i_b(wavelengths[wl_mask])
    E0_res = resampling.resample_E0(wavelengths[wl_mask])
    a_oz_res = resampling.resample_a_oz(wavelengths[wl_mask])
    a_ox_res = resampling.resample_a_ox(wavelengths[wl_mask])
    a_wv_res = resampling.resample_a_wv(wavelengths[wl_mask])
    da_w_div_dT_res = resampling.resample_da_w_div_dT(wavelengths[wl_mask])
    Ed_res = downwelling_irradiance.Ed(wavelengths[wl_mask])
    Ed_sa_res = downwelling_irradiance.Ed_sa(wavelengths[wl_mask])
    Ed_sr_res = downwelling_irradiance.Ed_sr(wavelengths[wl_mask])
    Ed_d_res = downwelling_irradiance.Ed_d(wavelengths[wl_mask])

    # Get params from file
    params = read_params(args.params_file)
    
    # Access xarray internal rasterio dataset reader
    dataset_reader = img.rio._manager.acquire()

    # Copy profile and change accordingly
    profile = dataset_reader.profile
    profile['driver'] = args.output_format
    profile['interleave'] = args.interleave
    profile['dtype'] = args.dtype

    # Create empty arrays for each output to write data in and 
    # prepare individual profiles for each of the outputs
    out_params = np.zeros((len(np.array(params)), img.shape[2])) * np.nan
    params_profile = profile.copy()
    params_profile['count'] = out_params.shape[0]

    out_nfev = np.zeros((1, img.shape[2])) * np.nan
    nfev_profile = profile.copy()
    nfev_profile['count'] = 1

    out_fwd = np.zeros((len(wavelengths[wl_mask]), img.shape[2])) * np.nan     
    fwd_profile = profile.copy()
    fwd_profile['count'] = len(wavelengths[wl_mask])

    out_glint = np.zeros((len(wavelengths[wl_mask]), img.shape[2])) * np.nan
    glint_profile = profile.copy()
    glint_profile['count'] = len(wavelengths[wl_mask])

    out_Rrs = np.zeros((len(wavelengths[wl_mask]), img.shape[2])) * np.nan
    Rrs_profile = profile.copy()
    Rrs_profile['count'] = len(wavelengths[wl_mask])

    ######################################################################
    ########## INVERSION LOOP ############################################
    ######################################################################

    # Create and open output files
    with rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_params']), 'w', **params_profile) as dst_params, \
         rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_nfev']), 'w', **nfev_profile) as dst_nfev, \
         rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_fwd']), 'w', **fwd_profile) as dst_fwd, \
         rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_glint']), 'w', **glint_profile) as dst_glint, \
         rio.open(('/').join([output_dir, args.input_file.split('/')[-1] + '_Rrs']), 'w', **Rrs_profile) as dst_Rrs:

            start = timeit.default_timer()
            
            # Loop over rows of input file, compute mask and write to output file        
            for idx, wind in dataset_reader.block_windows():
                print(str(idx[0]) + ' / ' + str(img.shape[1]) + ' : ' + str(wind))

                # Read row from xarray object
                row = img[:,idx[0],:]

                # Apply water mask and convert from reflectance R [-] to above-water radiance reflectance rrs [sr-1]
                rrs = row.where(indices.awei(row, row.wavelength)>0, shade=args.shade) / np.pi
                
                # Select only water pixels
                data = rrs[wl_mask].where(rrs.notnull().all(axis=0), drop=True).T

                # If the row contains water pixels ...
                if data.shape[0] > 0:
                    # Split the row into chunks ...
                    num_chunks = data.shape[0]  # Number of chunks to split the array into # 128

                    # ... each pixel is one chunk
                    chunk_size = len(data) // num_chunks
                    # Split the array of water pixels into chunks
                    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]  

                    # Put the chunks into the object store
                    chunk_refs = [ray.put(chunk) for chunk in chunks]  
                    # Invert chunks in parallel
                    result_refs = [invert_chunk.remote(chunk_ref,
                                                       params=params,
                                                       wavelengths=wavelengths[wl_mask],
                                                       weights=[],
                                                       a_i_spec_res=a_i_spec_res,
                                                       a_w_res=a_w_res,
                                                       b_phy_norm_res=b_phy_norm_res,
                                                       bb_w_res=bb_w_res,
                                                       R_i_b_res=R_i_b_res,
                                                       E0_res = E0_res,
                                                       a_oz_res = a_oz_res,
                                                       a_ox_res = a_ox_res,
                                                       a_wv_res = a_wv_res, 
                                                       da_w_div_dT_res = da_w_div_dT_res,
                                                       Ed_res = Ed_res,
                                                       Ed_sa_res = Ed_sa_res,
                                                       Ed_sr_res = Ed_sr_res,
                                                       Ed_d_res = Ed_d_res,
                                                       method='least_squares', 
                                                       max_nfev=1500) for chunk_ref in chunk_refs]  # Process the chunks in parallel
                    chunk_results = ray.get(result_refs) 

                    # Concatenate the results (MinimizerResult objects) from the processed chunks
                    results = np.concatenate(chunk_results)

                    ######################################################################
                    ########## TRANSLATE RESULTS INTO ARRAYS #############################
                    ######################################################################

                    ## params
                    # reset array so that results of laster iteration are set to nan
                    out_params = out_params * np.nan 
                    out_params[:,~np.isnan(rrs).all(axis=0)] = np.array([np.array(i.params) for i in results]).T
                    
                    ## n iterations
                    # reset array so that results of laster iteration are set to nan
                    out_nfev = out_nfev * np.nan
                    out_nfev[:,~np.isnan(rrs).all(axis=0)] = np.array([i.nfev for i in results]).T

                    ## forward rrs
                    # reset array so that results of laster iteration are set to nan
                    out_fwd = out_fwd * np.nan
                    out_fwd[:,~np.isnan(rrs).all(axis=0)] = np.array([model.forward(i.params,
                                                        wavelengths[wl_mask], 
                                                        a_i_spec_res=a_i_spec_res,
                                                        a_w_res=a_w_res,
                                                        b_phy_norm_res=b_phy_norm_res,
                                                        bb_w_res=bb_w_res,
                                                        R_i_b_res=R_i_b_res,
                                                        E0_res = E0_res,
                                                        a_oz_res = a_oz_res,
                                                        a_ox_res = a_ox_res,
                                                        a_wv_res = a_wv_res, 
                                                        da_w_div_dT_res = da_w_div_dT_res,
                                                        Ed_res = Ed_res,
                                                        Ed_sa_res = Ed_sa_res,
                                                        Ed_sr_res = Ed_sr_res,
                                                        Ed_d_res = Ed_d_res) 
                                            for i in results]).T
                    
                    ## glint
                    glint = np.array([surface.Rrs_surf(wavelengths[wl_mask], 
                                                                g_dd=i.params['g_dd'].value,
                                                                g_dsr=i.params['g_dsr'].value,
                                                                g_dsa=i.params['g_dsa'].value,
                                                                E0_res = E0_res,
                                                                a_ox_res = a_ox_res,
                                                                a_wv_res = a_wv_res, 
                                                                Ed_res = Ed_res,
                                                                Ed_sa_res = Ed_sa_res,
                                                                Ed_sr_res = Ed_sr_res,
                                                                Ed_d_res = Ed_d_res)
                                            for i in results])
                    # reset array so that results of laster iteration are set to nan
                    out_glint = out_glint * np.nan 
                    out_glint[:,~np.isnan(rrs).all(axis=0)] = glint.T

                    ## Rrs  
                    # reset array so that results of laster iteration are set to nan
                    out_Rrs = out_Rrs * np.nan 
                    out_Rrs[:, ~np.isnan(rrs).all(axis=0)] = (data - glint).T

                # Write output rows into the respective files
                dst_params.write(out_params[:,np.newaxis,:], window=wind)
                dst_nfev.write(out_nfev[:,np.newaxis,:], window=wind)
                dst_fwd.write(out_fwd[:,np.newaxis,:], window=wind)
                dst_glint.write(out_glint[:,np.newaxis,:], window=wind)
                dst_Rrs.write(out_Rrs[:,np.newaxis,:], window=wind)
                
            stop = timeit.default_timer()
            print('Processing time: ', stop - start, '')  

if __name__ == "__main__":
    main()
