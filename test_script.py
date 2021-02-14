import SUITPy as suit
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(base_dir,'functionalMapsSUIT'))

data = suit.flatmap.vol_to_surf(['MDTB08_Math.nii'])
suit.flatmap.plot(data)
pass
