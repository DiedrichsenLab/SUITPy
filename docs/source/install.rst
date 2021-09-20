Installation
============

Dependencies
------------

The required dependencies to use the software are:

* Python >= 3.6,
* setuptools
* Numpy >= 1.16
* Nibabel >= 2.5
* Pandas >= 0.24
* matplotlib >= 1.5.1

If you want to run the tests, you need pytest >= 3.9 and pytest-cov for coverage reporting.

Install over pip
----------------

First make sure you have installed all the dependencies listed above.
Then you can install SUITPy by running the following command in
a command prompt::

    pip install -U --user SUITPy

Install for developers
---------------------- 

Alternatively you can also fork or clone the repository at http://github.com/diedrichsenlab/SUITPy to a desired location (DIR). Simply include the lines::

    PYTHONPATH=/DIR/PcmPy:${PYTHONPATH}
    export PYTHONPATH 

To your ``.bash.profile`` or other shell startup file. 