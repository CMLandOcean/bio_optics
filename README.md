# bio_optics

The bio_optics python package (v0.0.1) is an open-source framework for forward and inverse modelling of above-water multi- and hyperspectral observations of natural water bodies. It contains the bio-optical models of Albert & Mobley (2003)[^1] and couples them to the surface reflectance model of Gege (2012)[^2]. The intention of the authors was to build a modular and extendable software package that allows the combination and creation of different models to study optical properties of natural water bodies. It can be used for simulation and analysis of spectroscopy data through inversion of the above-mentioned models. Single parts of the models can also be run as single modules, e.g., for the analysis of spectral backscattering or absorption measurements, or the correction of sun and sky glint at the water surface. 

### PLEASE GIVE US CREDIT
This python package has been created by members of the CarbonMapper Land and Ocean Program at the Center for Global Discovery and Conservation Science at Arizona State University. When using this framework, please use the following citation:

> _König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics_

### ACKNOWLEDGEMENTS
This repository is a framework for bio-optical modelling and builds on the extensive work of many researchers. It has mainly been inspired by the work of Peter Gege and the Water Color Simulator (WASI)[^3][^4][^5][^6]. Please give proper attribution when using this toolbox.

### LITERATURE
[^1]: Albert & Mobley (2003): An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. [doi.org/10.1364/OE.11.002873](https://doi.org/10.1364/OE.11.002873)

[^2]: Gege (2012): Analytic model for the direct and diffuse components of downwelling spectral irradiance in water. [10.1364/AO.51.001407](https://doi.org/10.1364/AO.51.001407)

[^3]: Gege (2004): The water color simulator WASI: an integrating software tool for analysis and simulation of optical in situ spectra. [10.1016/j.cageo.2004.03.005](https://doi.org/10.1016/j.cageo.2004.03.005)

[^4]: Gege & Albert (2006): A Tool for Inverse Modeling of Spectral Measurements in Deep and Shallow Waters. [https://doi.org/10.1007/1-4020-3968-9_4](https://doi.org/10.1007/1-4020-3968-9_4)

[^5]: Gege (2014): WASI-2D: A software tool for regionally optimized analysis of imaging spectrometer data from deep and shallow waters. [https://doi.org/10.1016/j.cageo.2013.07.022](https://doi.org/10.1016/j.cageo.2013.07.022)

[^6]: Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
