import os.path
import numpy as np
import numpy.typing as npt


from .grav_utils import read_icgem_coeff
from .grav_utils import get_potential


#TODO could add data directory structure for loading coeff if dir is passed
#TODO typing
#TODO do we want print statements to say whats happening

class GravPot():
    """Class to derive gravatational potential for a set of ECEF coordinates.
    """
    def __init__(self, r: npt.ArrayLike=np.array(6771.0,ndmin=1),
                 lat: npt.ArrayLike=np.array(0,ndmin=1),
                 lon: npt.ArrayLike=np.array(np.pi,ndmin=1),
                 pot_file: str = 'EIGEN-2.gfc',
                 lmax: int=50):
        """Ititialize the GravPot Class

        Position (r, lat, lon) needs to be in an Earth Centered Earth Fixed
        (ECEF) coordinate frame.     
        Units are m and radians for position (r, lat, lon).
        The Gravatational Potential is calculated in units of km^2/s^2.

        Parameters
        ----------
        r : npt.ArrayLike, optional
            An array of radial positions (ECEF - kilometers), 
            by default np.array(6771000.0,ndmin=1)
        lat : npt.ArrayLike, optional
            An array of latitude positions (ECEF - radians), 
            by default np.array(0,ndmin=1)
        lon : npt.ArrayLike, optional
            An array of longitude positions (ECEF - radians), 
            by default np.array(np.pi,ndmin=1)
        pot_file : str, optional
            ICGEM potential file used to derive the gravatational
            potential, uses .grav_utils.read_icgem_coeff which returns
            clm, slm, and metadata
            by default 'EIGEN-2.gfc'
        lmax : int, optional
            Maximum degree/order for deriving the gravatational potential. 
            If lmax is larger then the max degree/order of the loaded potential
            file (lload) then lmax is set to lload, 
            by default 50
        """
        r = np.asarray(r)
        if r.ndim == 0: r = r.reshape(1)
        lat = np.asarray(lat)
        if lat.ndim == 0: lat = lat.reshape(1)
        lon = np.asarray(lon)
        if lon.ndim == 0: lon = lon.reshape(1)

        # Initial Variables For Potential
        self.r = r
        self.lat = lat
        self.lon = lon
        
        self.pot_file = pot_file
        self.lmax = lmax
        self.gravpot = None

        # Modeled Field Variables Which are Loaded
        self.clm = None
        self.slm = None
        self.r0 = None
        self.GM = None

        if os.path.isfile(pot_file):
            self.load_coeff()

    def load_coeff(self, pot_file: str =None):
        """Load ICGEM clm, slm potential coefficients.

        Parameters
        ----------
        pot_file : str, optional
            ICGEM potential file used to derive the gravatational
            potential, uses .grav_utils.read_icgem_coeff which returns
            clm, slm, and metadata, by default None
        """        
        # potential file to use
        if pot_file:
            self.pot_file = pot_file

        self.clm, self.slm, cs_meta = read_icgem_coeff(self.pot_file)
        self.r0 = cs_meta['r0']/1000. # convert to km
        self.GM = cs_meta['GM']/(1000.**3) # convert to km^3/s^2
        # check the lmax of the potential file
        if not self.lmax:
            self.lmax = cs_meta['lmax']
        elif self.lmax > cs_meta['lmax']:
            print('Defined lmax is to large.')
            print('Setting lmax to that in the potential file.')
            self.lmax = cs_meta['lmax']

    def calc_pot(self):
        """Derive Potential.

        Potential is derived using normalize Legendre Coefficients from
        pyshtools.

        .grav_utils.get_potential is used to initially derive all required
        values which are then passed to .grav_utils._get_potential_numba_core
        which uses numba and jit to improve the performace of the calculation.

        The Gravatational Potential is calculated in units of km^2/s^2.

        Raises
        ------
        ValueError
            If potential coeffecients haven't been loaded.
        ValueError
            If the size of the position values, r, lat, lon are not the same.
        """
        # check if coeffecients have been loaded
        if self.clm is None:     
            raise ValueError('No coeffecients have been loaded, use load_coef( )')
        # check to make r, lat, lon are the same size
        if len(self.r) != len(self.lat) or len(self.r) != len(self.lon):
            raise ValueError('r, lat, and lon must be same length')

        self.gravpot = np.array([get_potential(rr, rlat, rlon,
                    self.clm, self.slm, self.GM, self.r0, lmax=self.lmax)
                    for rr, rlat, rlon in zip(self.r,self.lat,self.lon)])

    def get_pot(self):
        """Return Potential.

        Returns
        -------
        np.array
            Array of potetial in km^2/s^2
        """
        if self.gravpot:
            return self.gravpot
        