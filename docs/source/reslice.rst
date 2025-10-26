Reslice Module
===============

The reslice module is used to reslice images from individual into atlas space. Currently, the normalization needs to be done using the MATLAB version of the [`SUIT toolbox <https://github.com/jdiedrichsen/suit>`_]. We assume that you have run ``suit_normalize_dartel.m`` [`see here for documentation <https://www.diedrichsenlab.org/imaging/suit_function.htm#norm_dartel>`_]. The reslice module requires the result of the normalization step as a non-linear deformation file. In Matlab you can produce this file using the following command:

.. code-block::

    suit_save_darteldef(<name>, 'wdir',<dir>)

Where ``<name>`` is the name of the original anatomical and ``dir`` the suit working directory (defaults to current directory). This function produces ``y_<name>_suitdef.nii`` as an output. 

This file can then be used in SUITPy to reslice anatomical and functional images into atlas space. 

.. toctree::

    notebooks/reslice_example
