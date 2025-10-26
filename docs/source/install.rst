Installation
============

Dependencies
------------

The required dependencies to use the software are:

* python >= 3.10,
* setuptools
* numpy >= 1.22.0
* nibabel >= 3.2.1
* pandas >= 2.0.0
* matplotlib >= 3.5.0
* plotly >= 5.10.0
* scipy >= 1.9.0
* neuroimagingtools >= 1.1.1
* antspyx >= 0.6.1
* torch >= 2.0.0

Install over pip
----------------

First make sure you have installed all the dependencies listed above.
Then you can install SUITPy by running the following command in
a command prompt::

    pip install -U --user SUITPy

Install for developers
----------------------

Alternatively you can also fork or clone the repository at http://github.com/diedrichsenlab/SUITPy to a desired location (DIR). Simply include the lines::

    PYTHONPATH=/DIR/SUITPy:${PYTHONPATH}
    export PYTHONPATH

To your ``.bash.profile`` or other shell startup file.