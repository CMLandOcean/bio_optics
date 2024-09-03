# -*- coding: utf-8 -*-
#  Copyright 2023 
#  Center for Global Discovery and Conservation Science, Arizona State University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
# Translated to Python by:
#  Marcel König, mkoenig3 AT asu.edu 
#
# WaterQuality
#  Code is provided to Planet, PBC as part of the CarbonMapper Land and Ocean Program.
#  It builds on the extensive work of many researchers. For example, models were developed  
#  by Albert & Mobley [1] and Gege [2]; the methodology was mainly developed 
#  by Gege [3,4,5] and Albert & Gege [6].
#
#  Please give proper attribution when using this code for publication:
#
#  König, M., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): WaterQuality python package (Version x) [Software]. Available from https://github.com/CMLandOcean/WaterQuality
#
# [1] Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [10.1364/OE.11.002873]
# [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407]
# [3] Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005]
# [4] Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [10.1016/j.cageo.2013.07.022]
# [5] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6. 
# [6] Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [10.1007/1-4020-3968-9_4]


import os
import numpy as np
import pandas as pd
from spectral import BandResampler
from scipy.interpolate import interp1d
from .. water import backscattering


# get absolute path to data folder
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

### Water

def resample_a_w(wavelengths = np.arange(400,800)):
    """
    Absorption coefficient of pure water [m-1] at a reference temperature of 20 degC 
    as a compilation from different sources as distributed with the Water Color Simulator 6 (WASI6) [1]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to resample the absorption coefficient of pure water to
    :return: absorption coefficient of pure water absorption resampled to input wavelengths
    """
    # read file
    a_w_db = pd.read_csv(os.path.join(data_dir, 'a_w.txt'), skiprows=14, sep='\t', usecols=[0,1])
    # resample to sensor bands
    band_resampler = BandResampler(a_w_db.wavelength_nm.values, wavelengths)
    a_w = band_resampler(a_w_db["a"])
    return a_w


def resample_da_w_div_dT(wavelengths = np.arange(400,800)):
    """
    Temperature gradient of pure water absorption [m-1  degC-1]
    after Roettgers et al. (2013) [1] as distributed with the Water Color Simulator 6 (WASI6) [2]

    [1] Roettgers et al. (2013): Pure water spectral absorption, scattering, and real part of refractive index model.
                                 Algorithm Theoretical Basis Document "The Water Optical Properties Processor (WOPP).
                                 Distribution: Marc Bouvet, ESA/ESRIN
                                 Revision 7, May 2013
    [2] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the temperature gradient of pure water absorption to
    :return: temperature gradient of pure water absorption resampled to input wavelengths
    """
    # read file
    da_W_div_dT_db = pd.read_csv(os.path.join(data_dir, 'daWdT.txt'), skiprows=9, sep='\t')
    # resample to sensor bands
    band_resampler = BandResampler(da_W_div_dT_db.wavelength_nm.values, wavelengths)
    da_W_div_dT = band_resampler(da_W_div_dT_db["daW/dT"])
    return da_W_div_dT


def resample_a_i_spec(wavelengths = np.arange(400,800)):
    """
    Specific absorption coefficients [m2 mg-1] of six phytoplankton types compiled from multiple sources
    as distributed with the Water Color Simulator 6 (WASI6) [1] 

    1. phytoplankton
    2. cryptophyta
    3. cyanobacteria
    4. diatoms
    5. dinoflagellates
    6. green algae

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of six phytoplankton types resampled to input wavelengths
    """
    # read file
    a_phyto_db = pd.read_csv(os.path.join(data_dir, 'a_phy_spec.txt'), skiprows=25, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths) 
    a_i_spec = band_resampler(np.asarray(a_phyto_db)[:,1:])
    
    return a_i_spec


def resample_a_i_spec_EnSAD(wavelengths = np.arange(400,720)):
    """
    Specific absorption coefficients [m2 mg-1] of eight phytoplankton types from the supplemental data of [1] and WASI [2].

    1. Brown group
    2. Green group
    3. Cryptophyte
    4. Cyanobacteria blue
    5. Cyanobacteria red
    6. Coccolithophore
    7. Dinoflagellates from [2]
    8. Phytoplankton Case 1

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352] 
    [2] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of seven phytoplankton types resampled to input wavelengths
    """
    # read file
    a_phyto_db = pd.read_csv(os.path.join(data_dir, 'a_phy_spec_EnSAD.txt'), skiprows=11, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths) 
    a_i_spec = band_resampler(np.asarray(a_phyto_db)[:,1:])
    
    return a_i_spec


def resample_b_i_spec_EnSAD(wavelengths = np.arange(400,720)):
    """
    Specific scattering coefficients [m2 mg-1] of seven phytoplankton types from the supplemental data of [1].

    1. Brown group
    2. Green group
    3. Cryptophyte
    4. Cyanobacteria blue
    5. Cyanobacteria red
    6. Coccolithophore
    7. Dinoflagellates (identical to Brown group)
    8. Phytoplankton Case-1

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352] 

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of seven phytoplankton types resampled to input wavelengths
    """
    # read file
    b_phyto_db = pd.read_csv(os.path.join(data_dir, 'b_phy_spec_EnSAD.txt'), skiprows=4, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(b_phyto_db.wavelength_nm.values, wavelengths) 
    b_i_spec = band_resampler(np.asarray(b_phyto_db)[:,1:])
    
    return b_i_spec


def resample_bb_w(wavelengths = np.arange(400,800), 
                  fresh=True):
    """
    Spectral backscattering coefficient of water [m-1] at selected wavelengths according to Morel (1974) [1].

    [1] Morel (1974): Optical properties of pure water and pure Sea water.

    :param wavelengths: wavelengths to compute spectral backscattering coefficient of water for
    :param fresh:  boolean to decide if backscattering coefficient is to be computed for fresh (True, default) or oceanic water (False) with a salinity of 35-38 per mille. Values are only valid of lambda_0==500 nm.
    :return: spectral backscattering coefficient of water for input wavelengths
    """
    bb_w = backscattering.morel(wavelengths=wavelengths, fresh=fresh)
    
    return bb_w


def resample_b_phy_norm(wavelengths = np.arange(400,800)):
    """
    Normalized backscattering coefficient of phytoplankton as distributed with the Water Color Simulator 6 (WASI6) [1]
    obtained by fitting a measurement of b_b_phy(lambda) for green algae from Lake Garda in the range from 400 to 900 nm (Giardino, personal communication) 
    and extrapolating the fit curve to the range from 350 to 1000 nm.

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute normalized backscattering coefficient of phytoplankton for
    :return: normalized backscattering coefficient of phytoplankton for input wavelengths
    """
    # READ DATA FROM DATABASE
    b_phy_norm = pd.read_csv(os.path.join(data_dir, 'b_phy_norm.txt'), skiprows=6, sep="\t")
    # RESAMPLE TO WAVELENGTHS
    band_resampler = BandResampler(b_phy_norm.wavelength_nm.values, wavelengths)
    b_phy_norm = band_resampler(b_phy_norm["bb_phy_norm"])
    
    return b_phy_norm


def resample_R_i_b(wavelengths=np.arange(400,800)):
    """
    Irradiance reflectance or albedo [-] of bottom types f0..f5 to sensor's spectral sampling rate.

    0. constant 10% 
    1. sand 
    2. coral 
    3. cca (crustose coralline algae) 
    4. macrophyte
    5. seagrass (Zostera marina) 

    :param wavelengths: wavelengths to resample benthic substrate albedo to
    :return: irradiance reflectance or albedo of six benthic substrate types resampled to input wavelengths
    """
    # read file
    # R_bottom_db = pd.read_csv("C://Users//mkoenig3//WASI6//DATA/R_bottom.csv", sep=",")
    R_bottom_db = pd.read_csv(os.path.join(data_dir, 'R_b.txt'), skiprows=16, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(R_bottom_db.wavelength_nm.values, wavelengths)    
    R_i_b = band_resampler(np.asarray(R_bottom_db)[:,1:])
    
    return R_i_b
    
    
### Atmosphere

def resample_a_oz(wavelengths = np.arange(400,800)):
    """
    Absorption of ozone [cm-1] as distributed with the Water Color Simulator 6 (WASI6) [1]
    calculated using the radiative transfer model MODTRAN-3 (Gege 2012) [2]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spec- tral irradiance in water. [10.1364/AO.51.001407]

    :param wavelengths: wavelengths to compute ozone absorption coefficient for
    :return: ozone absorption coefficient for input wavelengths
    """
    # read file
    a_ozone_db = pd.read_csv(os.path.join(data_dir, 'a_ozone.txt'), sep="\t", skiprows=8)
    # resample to sensor bands
    band_resampler = BandResampler(a_ozone_db.wavelength_nm.values, wavelengths) 
    
    a_ozone_res = band_resampler(a_ozone_db["a"])    
    return a_ozone_res


def resample_a_wv(wavelengths = np.arange(400,800)):
    """
    Absorption of water vapour [cm-1] as distributed with the Water Color Simulator 6 (WASI6) [1]
    calculated using the radiative transfer model MODTRAN-3 (Gege 2012) [2]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spec- tral irradiance in water. [10.1364/AO.51.001407]

    :param wavelengths: wavelengths to compute water vapour absorption coefficient for
    :return: water vapour absorption coefficient for input wavelengths
    """
    # read file
    a_wv_db = pd.read_csv(os.path.join(data_dir, 'a_wv.txt'), sep="\t", skiprows=8)
    # resample to sensor bands
    band_resampler = BandResampler(a_wv_db.wavelength_nm.values, wavelengths) 
    
    a_wv_res = band_resampler(a_wv_db["a"])    
    return a_wv_res
    
    
def resample_a_ox(wavelengths = np.arange(400,800)):
    """
    Absorption of oxygen [cm-1] as distributed with the Water Color Simulator 6 (WASI6) [1]
    calculated using the radiative transfer model MODTRAN-3 (Gege 2012) [2]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Gege (2012): Analytic model for the direct and diffuse components of downwelling spec- tral irradiance in water. [10.1364/AO.51.001407]

    :param wavelengths: wavelengths to compute oxygen absorption coefficient for
    :return: oxygen absorption coefficient for input wavelengths
    """
    # read file
    a_ox_db = pd.read_csv(os.path.join(data_dir, 'a_oxygen.txt'), sep="\t", skiprows=8)
    # resample to sensor bands
    band_resampler = BandResampler(a_ox_db.wavelength_nm.values, wavelengths) 
    
    a_ox_res = band_resampler(a_ox_db["a"])    
    return a_ox_res
    
    
def resample_E0(wavelengths = np.arange(400,800)):
    """
    Extraterrestrial solar irradiance [mW m-2 nm-1] as distributed with the Water Color Simulator 6 (WASI6) [1]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to compute extraterrestrial solar irradiance for
    :return: extraterrestrial solar irradiance for input wavelengths
    """
    # read file
    E0_db = pd.read_csv(os.path.join(data_dir, 'E0_sun.txt'), sep=" ", skiprows=12)
    # resample to sensor bands
    band_resampler = BandResampler(E0_db.wavelength_nm.values, wavelengths) 
    
    E0_res = band_resampler(E0_db["E_0"])    
    return E0_res


### Other

def resample_n(wavelengths = np.arange(400,800)):
    """
    Real part of the refractive index of liquid water at a reference temperature of 25 degC after Segelstein [1] from the refractiveindex.info database.

    [1] Segelstein (1981): The complex refractive index of water. Master thesis. University of Missouri. Kansas City, MO. 

    Args:
        wavelengths: wavelengths to compute real part of the refractive index of water for
    Returns:
        real part of the refractive index of water for input wavelengths
    """
    n_db = pd.read_csv(os.path.join(data_dir, 'water_complex_refractive_index.txt'), sep=" ", skiprows=8, skipinitialspace=True).iloc[:-4].reset_index()
    n_db.columns = ['lambda', 'n', 'k']
    n_db = n_db.astype("float")
    # select wavelength region between 340 nm and 2500 nm
    n_db = n_db[(n_db["lambda"]>0.34) & (n_db["lambda"]<2.5)]
    # resample to target wavelengths
    band_resampler = BandResampler(n_db["lambda"].values*1000, wavelengths)   

    n_res = band_resampler(n_db["n"])
    return n_res


def resample_A(wavelengths = np.arange(400,800)):
    """
    Spectral parameters for the empirical a_Phy() simulation after Lee [1,2].
 
    [1] Lee et al. (1998): Hyperspectral remote sensing for shallow waters: 1 A semianalytical model [10.1364/AO.37.006329]
    [2] Lee (1994): VISIBLE-INFRARED REMOTE-SENSING MODEL AND APPLICATIONS FOR OCEAN WATERS. Dissertation.

    Args:
        wavelengths: wavelengths to empirical parameters A0 and A1 water for

    Returns:
        empirical parameters A0 and A1 for input wavelengths
    """
 
    # read file
    a = pd.read_csv(os.path.join(data_dir, 'a_phy_empirical_factors.txt'), sep='\t', skiprows=7)
    # resample to sensor bands
    band_resampler = BandResampler(a["wavelength_nm"].values, wavelengths)    
    a_0 = band_resampler(np.asarray(a["a_0"]))
    a_1 = band_resampler(np.asarray(a["a_1"]))
    
    return a_0, a_1


### Generic

def resample_spectra(spectra, in_wavelengths, out_wavelengths):
    """
    Generic resampling function
    """    
    band_resampler = BandResampler(in_wavelengths, out_wavelengths)
    resampled_spectra = band_resampler(spectra)

    return resampled_spectra


def resample_srf(srf_wavelengths, srf_factors, input_wavelengths, input_spectrum, kind='slinear', fill_value='extrapolate'):
    """
    Resample a spectrum to a sensor's band setting using it's spectral response function (SRF).
    Uses scipy.interpolate.interp1d.
    
    :param srf_wavelengths: wavelengths of srf [nm]
    :param srf_factors: spectral response factor per wavelength per band with shape (srf_wavelengths, n_output_bands)
    :param input_wavelengths: wavelengths of input spectrum [nm]
    :param input_spectrum: spectrum to be resampled with shape (n_bands) [1D], (n_bands, x) [2D] or (n_bands, x, y) [3D] 
    :param kind: specifies kind of interpolation, parameter for interp1d, default: 'slinear'
    :param fill_value: parameter for interp1d, default: 'extrapolate'
    :return: resampled spectrum with new band setting
    """
    # prepare empty array to fill, array shape depends on input shape
    resampled_spectrum = np.zeros((srf_factors.shape[1],) + input_spectrum.shape[1:])*np.nan
    
    for band_i in range(srf_factors.shape[1]):
        # fit interpolated SRF for respective band
        interp = interp1d(srf_wavelengths, srf_factors[:,band_i], kind=kind, fill_value=fill_value)
        interp_srf_factors = interp(input_wavelengths)
        
        # interpolate original spectrum to SRF bands, multiply interpolated SRF with spectrum, sum and divide by sum of SRF
        # if input is 1D
        if len(input_spectrum.shape)==1:
            resampled_spectrum[band_i] = np.einsum('i,i->i', interp(input_wavelengths), input_spectrum).sum(axis=0) / np.sum(srf_factors[:,band_i])
        # else if input is 2D
        elif len(input_spectrum.shape)==2:
            resampled_spectrum[band_i,:] = np.einsum('i,ik->ik', interp_srf_factors, input_spectrum).sum(axis=0) / np.sum(srf_factors[:,band_i])
        # else if input is 3D
        elif len(input_spectrum.shape)==3:
            resampled_spectrum[band_i,:] = np.einsum('i,ijk->ijk', interp_srf_factors, input_spectrum).sum(axis=0) / np.sum(srf_factors[:,band_i])
        
    return resampled_spectrum