import numpy as np
from .. import fluorescence
from . import bi


def forward(parameters,
            wavelengths,
            h_C_res=None,
            h_C_phycocyanin_res=None,
            h_C_phycoerythrin_res=None,
            **kwargs):
    """
    HEREON water model (Bi et al. 2023) with chlorophyll-a, phycocyanin, and
    phycoerythrin fluorescence terms.

    Calls ``bi.forward`` for the IOP-based Rrs, then conditionally adds:

    - Chl-a fluorescence (``fluorescence.Rrs_fl``) when ``sum(C_0..C_7) > 0.1``
    - Phycocyanin fluorescence (``fluorescence.Rrs_fl_phycocyanin``) when ``C_3 > 0.1``
    - Phycoerythrin fluorescence (``fluorescence.Rrs_fl_phycoerythrin``) when ``C_4 > 0.1``

    Additional fluorescence parameters required in *parameters*:

    - ``L_fl_lambda0``, ``W``, ``fwhm1``, ``fwhm2``, ``lambda_C1``, ``lambda_C2``, ``double``
    - ``L_fl_phycocyanin``, ``fwhm_phycocyanin``, ``lambda_C_phycocyanin``  (only needed when C_3 > 0.1)
    - ``L_fl_phycoerythrin``, ``fwhm_phycoerythrin``, ``lambda_C_phycoerythrin``  (only needed when C_4 > 0.1)

    All other parameters and precomputed cache kwargs are forwarded to ``bi.forward``.

    Args:
        parameters: lmfit Parameters object
        wavelengths: wavelengths [nm]
        h_C_res: optional precomputed Chl-a fluorescence lineshape
        h_C_phycocyanin_res: optional precomputed phycocyanin fluorescence lineshape
        h_C_phycoerythrin_res: optional precomputed phycoerythrin fluorescence lineshape
        **kwargs: forwarded to ``bi.forward`` (all precomputed cache arguments)

    Returns:
        Rrs_sim: above-water remote sensing reflectance [sr-1]
    """
    R_rs = bi.forward(parameters, wavelengths, **kwargs)

    C_phy = sum(float(parameters[f"C_{i}"]) for i in range(8))

    if C_phy > 0.1:
        R_rs = R_rs + fluorescence.Rrs_fl(
            wavelengths=wavelengths,
            L_fl_lambda0=parameters['L_fl_lambda0'],
            W=parameters['W'],
            fwhm1=parameters['fwhm1'],
            fwhm2=parameters['fwhm2'],
            lambda_C1=parameters['lambda_C1'],
            lambda_C2=parameters['lambda_C2'],
            double=parameters['double'],
            h_C_res=h_C_res,
        )

    if float(parameters["C_3"]) > 0.1:
        R_rs = R_rs + fluorescence.Rrs_fl_phycocyanin(
            wavelengths=wavelengths,
            L_fl_phycocyanin=parameters['L_fl_phycocyanin'],
            fwhm=parameters['fwhm_phycocyanin'],
            lambda_C=parameters['lambda_C_phycocyanin'],
            h_C_phycocyanin_res=h_C_phycocyanin_res,
        )

    if float(parameters["C_4"]) > 0.1:
        R_rs = R_rs + fluorescence.Rrs_fl_phycoerythrin(
            wavelengths=wavelengths,
            L_fl_phycoerythrin=parameters['L_fl_phycoerythrin'],
            fwhm=parameters['fwhm_phycoerythrin'],
            lambda_C=parameters['lambda_C_phycoerythrin'],
            h_C_phycoerythrin_res=h_C_phycoerythrin_res,
        )

    return R_rs
