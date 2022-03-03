Reslice Module
===============

The reslice module of the SUITPy toolbox is used to reslice images from individual into Atlas space. Currently, we are still relying on the noramlization being done in the [`noramlization function <https://github.com/jdiedrichsen/suit/blob/master/suit_normalize_dartel.m>`_](GITHUB) of the toolbox. We assume that you have `run suit_normalize_dartel.m` (see [`Documentation of the normalization function <http://www.diedrichsenlab.org/imaging/suit_function.htm#norm_dentate>`_]). The reslice module requires the non-linear deformation from the normalize steo as a nonlinear deformation file. In Matlab you can produce this file using the following command:
```
[Def, mat] = spmdefs_get_dartel(flowfield, affine_matrix)
```
Also, the non-linear deformation file can be saved by the following command:
```
spmdefs_save_def(Def, mat, 'suitdef')
```
The reslice toolbox has three necessary input: deformation (y_xxx.nii), interpolation (trilinear or nearest neighbor), and affine matrix. In addition, it allows users to input mask image that only show particular area. Also, voxel size and image shape can be defined by users.


.. toctree::

    notebooks/reslice_example
