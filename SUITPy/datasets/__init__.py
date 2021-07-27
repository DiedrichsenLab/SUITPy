"""
Helper functions to download NeuroImaging datasets
"""

from warnings import warn
from .struct import (fetch_icbm152_2009, load_mni152_template,
                     load_mni152_brain_mask, load_mni152_gm_template,
                     load_mni152_gm_mask, load_mni152_wm_template,
                     load_mni152_wm_mask, fetch_oasis_vbm,
                     fetch_icbm152_brain_gm_mask,
                     MNI152_FILE_PATH, GM_MNI152_FILE_PATH,
                     WM_MNI152_FILE_PATH, fetch_surf_fsaverage)
from .func import (fetch_haxby,
                   fetch_adhd, fetch_miyawaki2008,
                   fetch_localizer_contrasts, fetch_abide_pcp,
                   fetch_localizer_button_task,
                   fetch_localizer_calculation_task, fetch_mixed_gambles,
                   fetch_megatrawls_netmats, fetch_cobre,
                   fetch_surf_nki_enhanced, fetch_development_fmri,
                   fetch_language_localizer_demo_dataset,
                   fetch_bids_langloc_dataset,
                   fetch_openneuro_dataset_index,
                   select_from_index,
                   patch_openneuro_dataset,
                   fetch_openneuro_dataset,
                   fetch_localizer_first_level,
                   fetch_spm_auditory,
                   fetch_spm_multimodal_fmri,
                   fetch_fiac_first_level,
                   )
from .atlas import (fetch_atlas_schaefer_2018)

from .utils import get_data_dirs

__all__ = [
           ]

warn("This module is experimental", FutureWarning)
