from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='bio_optics',
    version='0.0.1',
    description='Bio-optical modelling for spectroscopy applications.',
    author='Marcel König',
    author_email='mkoenig3@asu.edu',
    url = 'https://github.com/CMLandOcean/bio_optics',
    keywords=['water', 'optics', 'bio-optical modelling', 'spectroscopy'],
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'bio_optics': ['data/*.txt']
    },
    license='Apache-2.0',
    install_requires=[
        'numpy >= 1.21.5',
        'scipy >= 1.7.3',
        'pandas >= 1.3.5',
        'spectral >= 0.22.4',
        'lmfit >= 1.0.3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'
    ],
)