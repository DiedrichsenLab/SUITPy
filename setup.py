#! /usr/bin/env python

"""
@author: maedbhking
based heavily on flexible functionality of nilearn `setup.py`
"""

descr = """A python package for cerebellar neuroimaging..."""

import sys
import os

from setuptools import setup, find_packages

def load_version():
    """Executes SUITPy/version.py in a globals dictionary and return it.
    Note: importing SUITPy is not an option because there may be
    dependencies like nibabel which are not installed and
    setup.py is supposed to install them.
    """
    # load all vars into globals, otherwise
    #   the later function call using global vars doesn't work.
    globals_dict = {}
    with open(os.path.join('SUITPy', 'version.py')) as fp:
        exec(fp.read(), globals_dict)

    return globals_dict


def is_installing():
    # Allow command-lines such as "python setup.py build install"
    install_commands = set(['install', 'develop'])
    return install_commands.intersection(set(sys.argv))


def list_required_packages():
    required_packages = []
    required_packages_orig = ['%s>=%s' % (mod, meta['min_version'])
                              for mod, meta
                              in _VERSION_GLOBALS['REQUIRED_MODULE_METADATA']
                              ]
    for package in required_packages_orig:
        required_packages.append(package)
    return required_packages


# Make sources available using relative paths from this file's directory.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_VERSION_GLOBALS = load_version()
DISTNAME = 'SUITPy'
DESCRIPTION = 'Mapping and plotting cerebellar fMRI data in Python'
with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()
MAINTAINER = 'Maedbh King'
MAINTAINER_EMAIL = 'maedbhking@berkeley.edu'
URL = 'https://github.com/DiedrichsenLab/SUITPy'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/DiedrichsenLab/SUITPy/archive/refs/tags/v1.0.4.tar.gz'
VERSION = _VERSION_GLOBALS['__version__']


if __name__ == "__main__":
    if is_installing():
        module_check_fn = _VERSION_GLOBALS['_check_module_dependencies']
        module_check_fn(is_SUITPy_installing=True)

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
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
          ],
          packages=find_packages(),
          package_data={
              'SUITPy': ['surfaces/*.surf.gii', 'surfaces/*.C.scene', 'surfaces/*.shape.gii', 'surfaces/*.txt'],
          },
          install_requires=list_required_packages(),
          python_requires='>=3.6',
          )
