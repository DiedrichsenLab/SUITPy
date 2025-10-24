#! /usr/bin/env python

"""
based heavily on flexible functionality of nilearn `setup.py`
"""

descr = """A python package for cerebellar neuroimaging."""

import sys
import os
from setuptools import setup, find_packages

_SUITPy_INSTALL_MSG = 'See %s for installation information.' % (
    'https://suitpy.readthedocs.io/en/latest/install.html#installation')

REQUIRED_MODULE_METADATA = ['numpy>=1.22.0',
                            'nibabel>=3.5.0',
                            'neuroimagingtools>=1.1.1',
                            'requests>=2',
                            'scipy>=1.9.0',
                            'matplotlib>=3.5.0',
                            'plotly>=5.3.1',
                            'pandas>=2.0.0',
                            'joblib>=1.2.0',
                            'scipy>=1.9.0',
                            'antspyx>=0.6.1',
                            'torch>=2.5.0',]

def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))

# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DESCRIPTION = 'Mapping and plotting cerebellar fMRI data in Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
DOWNLOAD_URL = 'https://github.com/DiedrichsenLab/SUITPy/archive/refs/tags/v.1.3.3.tar.gz'
VERSION = '1.3.3'

if __name__ == "__main__":

    setup(name='SUITPy',
          maintainer='Jorn Diedrichsen',
          maintainer_email='joern.diedrichsen@googlemail.com',
          description=DESCRIPTION,
          license='MIT',
          url='https://github.com/DiedrichsenLab/SUITPy',
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
              'Programming Language :: Python :: 3.10',
          ],
          packages=find_packages(),
          package_data={
              'SUITPy': ['surfaces/*.surf.gii',
                         'surfaces/*.C.scene',
                         'surfaces/*.shape.gii',
                         'surfaces/*.txt',
                         'templates/*.nii.gz',
                         'parameters/*.pkl'],
          },
          install_requires=REQUIRED_MODULE_METADATA,
          python_requires='>=3.10',
          )
