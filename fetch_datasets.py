#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import nibabel as nb
import warnings

base_dir = os.path.dirname(os.path.abspath(__file__))
surf_dir = os.path.join(_base_dir, 'surfaces')
atlas_dir = os.path.join(_base_dir, 'atlasesSUIT')
func_dir = os.path.join(_base_dir, 'functionalMapsSUIT')

    