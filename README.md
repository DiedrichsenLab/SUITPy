# SUITPy: A python toolbox for the analysis and visualization of cerebellar imaging data. 

The package provides some basic functionality of the original [SUIT toolbox for Matlab](https://github.com/jdiedrichsen/suit). Currently, only mapping of volume data to the flatmap and display of the flatmap are implemented. 

## Installation

* Unzip the archive (or clone directly) and place it in a good location
* Adjust your environmental variable `$PYTHONPATH` to include the folder that contains the SUITPy package. You can do this by adding the line 
```
PYTHONPATH=/<MY PATH>:${PYTHONPATH}
```
* ensure that you have the following 3 dependencies installled (these can be installed via pip or conda):
	* nibabel
	* numpy
	* matplotlib	 	 
* In python you can then import the package with 
```
import SUITPy as suit
```

## Usage 
For basic use of the toolbox for mapping and display of cerebellar imaging data is demonstrated in the Jupyter notebook `notebooks/flatmap_example.ipynb`. 

## Licence and Acknowledgements

The Pyhton version of the SUIT toolbox has been developed by J. Diedrichsen (joern.diedrichsen@googlemail.com), D. Zhi, C. Hernandez-Castillo, M. King, and many others. It is distributed under the [Creative Commons Attribution-NonCommercial 3.0 Unported License](http://creativecommons.org/licenses/by-nc/3.0/deed.en_US), meaning that it can be freely used for non-commercial purposes, as long as proper attribution in form of acknowledgments and links (for online use) or citations (in publications) are given. The relevant references are:

##### SUIT normalisation and template: 

- Diedrichsen, J. (2006). A spatially unbiased atlas template of the human cerebellum. *Neuroimage. 33(1)*, 127-138. 

##### Surface-based representation and flatmap

- Diedrichsen, J. & Zotow, E. (2015). Surface-based display of volume-averaged cerebellar data. *PLOSOne*. 

