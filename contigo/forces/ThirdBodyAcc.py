import pandas as pd
import numpy as np
import numpy.typing as npt

import spiceypy as spice

from .tba_utils import tba_pairwise_numba
from .constants import GMc

class ThirdBodyAcc():

    def __init__(self, 
                 spos: npt.ArrayLike=np.array([6771.0,0,0],ndmin=2),
                 stime: npt.NDArray[np.datetime64]=np.array(pd.to_datetime('2020-01-01')),
                 body: npt.ArrayLike=['SUN'],
                 GM: npt.ArratyLike | None = None,
                 scale: str | None = None,
                 ephemeris: str='de440s'):
    

        #need to furnish spice kernels here

        # allowed ephemeris to load
        allowed_eph = ['de440','de440s']
        if ephemeris in allowed_eph:
            eph_sh = ephemeris[0:5]

        # allowed time scales for getting third body accelerations
        scale = scale.upper()
        allowed_scales = ['GPS','TAI','UTC','ET','TDB']
        if not scale:
            raise ValueError(f'scale must be one of {allowed_scales}')
        elif scale not in allowed_scales:
            raise ValueError(f'Incorrect scale {scale}, scale must be one of {allowed_scales}')

        # calculate the seconds past j2000 to convert
        # to SPICE ET Ephemeris time (in the SPICE system,
        # this is equivalent to TDB time)
        j2000 = pd.Timestamp('2000-01-01 12:00:00')
        spj2000 = ((stime - j2000).dt.total_seconds()).to_list()

        # set all needed attributes
        self.et = [spice.unitim(sp_in,scale,'ET') for sp_in in spj2000]
        self.spos = spos
        self.body = [bd.upper for bd in body]
        self.stime = pd.to_datetime(stime)
        if not GM:
            self.GM = [GMc[eph_sh][bd] for bd in self.body]
        else:
            self.GM = GM

        #attributes used later
        self.bd_ecef = None
        self.bd_acc = None

    def calc_tba(self):

        # get the body positions in ecef
        bd_ecef = [spice.spkpos(bd,self.et,'ITR93','NONE','EARTH')[0]
                   for bd in self.body]
        
        bd_acc = tba_pairwise_numba(self.spos, bd_ecef, self.GM)

        self.bd_acc = bd_acc
        self.bd_ecef = bd_ecef

    def get_tba(self):
        return self.bd_acc
    
    def get_body_pos(self):
        return self.bd_ecef
