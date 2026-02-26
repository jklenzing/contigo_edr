
import posixpath
import urllib.parse
import logging


from os import path

import pandas as pd
import numpy as np
import numpy.typing as npt
import spiceypy as spice

import contigo.config as config


logger = logging.getLogger(__name__)

def spice_et(stime: npt.ArrayLike, tscale: str):

    allowed = {'GPS', 'TAI', 'UTC', 'ET', 'TDB'}
    tscale = tscale.upper()
    if tscale not in allowed:
        raise ValueError(f"tscale must be one of {allowed}")

    # make sure the leap second kernel is loaded
    # if it isn't check for it or download it and load it
    check_lpsk()

    if tscale == 'UTC':
        t_str = pd.to_datetime(np.array(stime)).strftime('%d %b %Y %H:%M:%S.%f')
        et = np.array([spice.utc2et(sp_in) for sp_in in t_str]) 
    else:
        j2000 = pd.Timestamp('2000-01-01 12:00:00')
        spj2000 = ((stime - j2000).total_seconds()).to_list()
        et = np.array([spice.unitim(sp_in,tscale,'ET') for sp_in in spj2000])

    return et

def check_lpsk( ):
    # get the leapsecond kernel and make sure it's
    # loaded, download it if we don't have it
    leaps_f = config.LEAP_FILE
    lp_kernel = path.join(config.DATA_DIR,leaps_f)

    sp_kcnt = spice.ktotal('ALL')
    sp_loaded = [spice.kdata(i,'ALL')[0] for i in range(sp_kcnt)]
    if path.exists(lp_kernel) and [lp_kernel] not in sp_loaded:
        spice.furnsh(lp_kernel) # need to check if kernels are loaded
    else:
        base_url = 'https://naif.jpl.nasa.gov'
        base_pth = '/pub/naif/generic_kernels'
        leaps_d = 'lsk'

        leaps_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,leaps_d,leaps_f))

        logger.info('Downloading kernel - %s', lp_kernel)
        dl_file(leaps_url, lp_kernel)
        spice.furnsh(lp_kernel)