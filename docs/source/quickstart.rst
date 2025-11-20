.. _quickstart:

Quick Start
===========
This tutorial assumes basic familiarity with Python and NIfTI images, but
**no prior experience with the Matlab SUIT toolbox is required**.
For background on the SUIT template and Matlab toolbox, see the  
`SUIT website <http://diedrichsenlab.org/imaging/suit.htm>`_.

Overview
~~~~~~~~

This tutorial introduces the basic workflow for conducting a group-level
cerebellar fMRI analysis using **SUITPy**. The aim is to take each
participant’s anatomical scan and functional contrast map, transform them
into SUIT space, and prepare the resulting data for group statistics and
visualisation.

For each subject, ensure that the following files are available:

* A high-quality **T1-weighted anatomical image**
  (e.g., ``sub-01_T1w.nii.gz``)
* One or more **functional contrast maps**, already coregistered to the
  T1 image  
  (e.g., ``sub-01_taskA_contrastB.nii.gz``)

We recommend to use a minimal processing pipeline, including only correction for spatial distortions, motion correction, and coregistration to the anatomical image. The first-level GLM is best calculated in the native space of the subject - normalization to MNI space or any smoothing is best avoided at this point. 

.. note::

   SUITPy can also be applied to data that have already been normalized
   to MNI space, provided that both the T1 image and the corresponding
   contrast maps are in that same space.  
   However, **smoothing should still be avoided** to prevent undesired
   cortical influence on cerebellar activation patterns.

Before You Start
----------------

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

To run SUITPy, ensure that you have:

* Python 3.10 or newer
* A virtual environment (``venv`` or ``conda``)
* Installed the packages listed in :doc:`install`
  (NumPy, nibabel, PyTorch, antspyx, neuroimagingtools, etc.)

Install SUITPy
--------------

If SUITPy is not installed, run:

.. code-block:: bash

   pip install -U --user SUITPy

For detailed instructions and dependencies, see :doc:`install`.

Your First SUITPy Analysis
--------------------------

This tutorial demonstrates:

1. Cerebellar isolation  
2. Normalization to SUIT template space  
3. Reslicing and mapping functional MRI  
4. Plotting results on the SUIT flatmap  

To make this tutorial reproducible and interactive, the full workflow
is provided as a Jupyter notebook.

Jupyter Notebook Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

You can access the full notebook from the link below and follow it step by step to quickly run a complete SUITPy workflow.

.. toctree::
   :maxdepth: 1

   tutorials/1.quickstart_fMRI.ipynb

Where to Go Next
----------------

After completing this quick start tutorial, consider exploring:

* :doc:`isolate` — details on cerebellar isolation models and options
* :doc:`normalize` — normalization of cerebellar and functional images
* :doc:`reslice` — reslicing functional data into cerebellar space
* :doc:`flatmap` — mapping volumes and labels onto a 2D flatmap
* :doc:`reference` — full API reference for SUITPy

For advanced workflows, examples, and troubleshooting, see the tutorials
folder.


