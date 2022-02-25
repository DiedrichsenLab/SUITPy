Reslice Module
===============

The reslice module of the SUITPy toolbox is used to reslice images from individual into Atlas space. Currently, we are still relying on the noramlization being done in the [Matlab version](GITHUB) of the toolbox. We assume that you have `run suit_normalize_dartel.m` (see [documentation](). The reslice module requires the non-linear deformation from the normalize steo as a nonlinear deformation file. In Matlab you can produce this file using the following commands: 

```
MATLAB COMMAND 
```

The reslice toolbox  (y_xxx.nii), interpolation (trilinear or nearest neighbor), and affine matrix. In addition, it allows users to input mask image that only show particular area. Also, voxel size and image shape can be defined by users.


.. toctree::

    notebooks/reslice_example
