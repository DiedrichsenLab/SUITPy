Isolation Module
================

The isolation module of the SUIT toolbox uses a pre-trained convolutional neural network to isolate the cerebellum and brainstem from the rest of the head. The network was trained on manually labelled anatomical images from a wide range of studies, scanners and acquisition protocols. It works more reliably and accurately than previous (Matlab) versions of the SUIT toolbox. 
The network is based on the U-Net architecture and implemented using the Pytorch package. The network was developed by Yao Li with supervision from Carlos Hernandez-Castillo and Jörn Diedrichsen.

Usage
-----

There are two ways to use the module

1. Import the isolate function from the SUITpy library and use directly (recommended)

.. toctree::
   :maxdepth: 2

   notebooks/isolate_example


2. Alternatively, you can run the script directly via the terminal or bash script. 
::
    python isolate.py --T1 T1w_image

The optional input parameters are 
::
    --T1: T1w- image  
    --T2: Additional (or standalone T2w image) 
    --result_folder dir: 
    --brain_mask Mask_image: Binary mask image from skull stripping step to improve affine normalization 
    --template template: Templa] [--params PARAMS] [--save_cropped_files] [--save_transform]
isolate.py: error: argument -h/--help: ignored explicit argument 'elp'


Architecture
------------
Here is the overall Unet architecture

.. image:: Unet.png

The model takes 2 input channels which are filled with T1w and T2w images respectively. 0 padding is used if any input channel is empty. The model would work on any single modality input (T1w/T2w) and have better performance when both modalities are provided.
If T1w and T2w images are fed, two images must be co-registered to each other.

