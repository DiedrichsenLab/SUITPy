import SUITPy as suit
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(base_dir))

# data = suit.flatmap.vol_to_surf(['functionalMapsSUIT/MDTB08_Math.nii'])
# suit.flatmap.plot(data,threshold = 0.05)
# labeldata = suit.flatmap.vol_to_surf(['atlasesSUIT/MDTB_10Regions.nii'],stats = 'mode')
# suit.flatmap.plot(labeldata, overlay_type='label',cmap='tab10')
suit.flatmap.plot('surfaces/Buckner_17Networks.label.gii',overlay_type='label')
