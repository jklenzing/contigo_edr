from .grav_utils import read_icgem_coeff
from .grav_utils import get_potential

import numpy as np
import numpy.typing as npt


#TODO could add data directory structure for loading coeff if dir is passed
#TODO typing

class GravPot():

    def __init__(self, r: npt.ArrayLike=[6771000.0], 
                 lat: npt.ArrayLike=[0], 
                 lon: npt.ArrayLike=[np.pi],
                 pot_file: str = 'EIGEN-2.gfc', 
                 lmax: int=None):
        
        r = np.asarray(r)
        if r.ndim == 0: r = r.reshape(1)
        lat = np.asarray(lat)
        if lat.ndim == 0: lat = lat.reshape(1)
        lon = np.asarray(lon)
        if lon.ndim == 0: lon = lon.reshape(1)

        self.r = r
        self.lat = lat
        self.lon = lon
        self.pot_file = pot_file
        self.lmax = lmax
    
    def load_coeff(self, pot_file: str =None):
        # potential file to use
        if pot_file:
            self.pot_file = pot_file

        self.clm, self.slm, cs_meta = read_icgem_coeff(self.pot_file)
        self.r0 = cs_meta['r0']
        self.GM = cs_meta['GM']
        # check the lmax of the potential file
        if not self.lmax:
            self.lmax = cs_meta['lmax']
        elif self.lmax > cs_meta['lmax']:
            print('Defined lmax is to large.')
            print('Setting lmax to that in the potential file.')
            self.lmax = cs_meta['lmax']

    def get_pot(self):
        if not hasattr(self, 'clm') or not hasattr(self, 'slm'):
            raise ValueError('No coeffecients have been loaded, use load_coef( )')
        
        if len(self.r) != len(self.lat) or len(self.r) != len(self.lon):
            raise ValueError('r, lat, and lon must be same length')
        
        self.gravpot = [get_potential(rr, rlat, rlon,
                    self.clm, self.slm, self.GM, self.r0, lmax=self.lmax)
                    for rr, rlat, rlon in zip(self.r,self.lat,self.lon)]
        