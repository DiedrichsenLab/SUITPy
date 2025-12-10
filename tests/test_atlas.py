#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: joerndiedrichsen
"""

import SUITPy as suit

def test_atlas_dir():
    dirs1 = suit.utils.get_atlas_dirs()
    dirs2 = suit.utils.get_atlas_dirs('~/data/cerebellar_atlases')
    pass

def test_find_atlas():
    dirs1 = suit.utils._get_atlas_dir('Buckner_2011',create_dir=False)
    dirs2 = suit.utils._get_atlas_dir('Buckner_test',create_dir=False)
    dirs3 = suit.utils._get_atlas_dir('Buckner_test',create_dir=True)
    pass

def test_fetch_atlas():
    suit.fetch_atlas('Diedrichsen_2009',atlas_dir='/Users/jdiedrichsen/cerebellum_test')

if __name__ == "__main__":
    # test other functions if needed
    test_fetch_atlas()
    pass
