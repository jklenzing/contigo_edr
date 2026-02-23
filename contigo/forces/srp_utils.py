"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 23/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import sys
from os import path

import contigo.config as config

def setup_gmat(apistartup, gmat_install):

    gmat_bin_path = gmat_install + "/bin"
    startup = gmat_bin_path + "/" + apistartup

    if path.exists(startup):
        sys.path.insert(1, gmat_bin_path)
        import gmatpy as gmat
        gmat.Setup(startup)

        config.state['gmat_loaded'] = True
    else:
        raise ValueError(f"Cannot find {startup} in {gmat_bin_path}")