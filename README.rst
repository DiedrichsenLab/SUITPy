
	.. -*- mode: rst -*-

.. image:: https://img.shields.io/pypi/v/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: Pypi Package

.. image:: https://img.shields.io/pypi/pyversions/nilearn.svg
    :target: https://pypi.org/project/nilearn/
    :alt: PyPI - Python Version

.. image:: https://github.com/nilearn/nilearn/workflows/build/badge.svg?branch=main&event=push
   :target: https://github.com/nilearn/nilearn/actions
   :alt: Github Actions Build Status

.. image:: https://codecov.io/gh/nilearn/nilearn/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/nilearn/nilearn
   :alt: Coverage Status

.. image:: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=main
   :target: https://dev.azure.com/Parietal/Nilearn/_apis/build/status/nilearn.nilearn?branchName=main
   :alt: Azure Build Status

SUITPy

Analysis and visualization of cerebellar imaging data.
=======

The package provides some basic functionality of the original SUIT toolbox for Matlab(https://github.com/jdiedrichsen/suit). 
Currently, only mapping of volume data to the flatmap and display of the flatmap are implemented. 

Important links
===============

- SUIT website: http://diedrichsenlab.org/imaging/SUIT.htm
- Official source code repository: https://github.com/SUITPy/
- HTML documentation (stable release): http://SUITPy.github.io/

Dependencies
============

The required dependencies to use the software are:

* Python >= 3.6,
* setuptools
* Numpy >= 1.16
* Nibabel >= 2.5
* Pandas >= 0.24
* matplotlib >= 1.5.1

If you want to run the tests, you need pytest >= 3.9 and pytest-cov for coverage reporting.

Install
=======

First make sure you have installed all the dependencies listed above.
Then you can install SUITPy by running the following command in
a command prompt::

    pip install -U --user SUITPy

More detailed instructions are available at
http://SUITPy.github.io/introduction.html#installation.

Development
===========

Detailed instructions on how to contribute are available at
http://SUITPy.github.io/development.html


Licence and Acknowledgements
=========================== 
The Python version of the SUIT toolbox has been developed by the Diedrichsenlab including J. Diedrichsen, M. King, D. Zhi, C. Hernandez-Castillo, S. Witt and others. It is distributed under MIT License, meaning that it can be freely used and re-used, as long as proper attribution in form of acknowledgments and links (for online use) or citations (in publications) are given. The relevant references are:

SUIT normalisation and template: 

- Diedrichsen, J. (2006). A spatially unbiased atlas template of the human cerebellum. Neuroimage. 33(1), 127-138. 

Surface-based representation and flatmap

- Diedrichsen, J. & Zotow, E. (2015). Surface-based display of volume-averaged cerebellar data. PLOSOne. 