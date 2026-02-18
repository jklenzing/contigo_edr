import posixpath
import urllib.parse
from pathlib import Path
from os import path, makedirs
from datetime import datetime
from dateutil import tz

import pandas as pd
import numpy as np
import numpy.typing as npt
import spiceypy as spice

from .tba_utils import tba_pairwise_numba
from .constants import GMc

from ..utils import dl_file, wf_mtime

class ThirdBodyAcc():
    """Deriving Third Body Acceleration using JPL SPICE
    """
    def __init__(self, 
                 spos: npt.ArrayLike=np.array([6771.0,0,0],ndmin=2),
                 stime: npt.ArrayLike=pd.Series(pd.to_datetime('2020-01-01')),
                 body: npt.ArrayLike=['SUN'],
                 GM: npt.ArrayLike | None = None,
                 scale: str | None = None,
                 ephemeris: str='de440s'):
        """Initialize the ThirdBodyAcc Class for deriving accelleration from third
        bodies such as the Sun and Moon.

        Position is a (n,3) array that needs to be in Earth Centered Earth Fixed
        (ECEF) coordinate frame. Units are km's. 
        
        The body GM constants are in units of km^3/s^2. 

        Uses the SPICE (Spacecraft, Planet, Instrument, C-matrix, Events) observation 
        geometry information system and Kernels from JPLs Navigation and Ancillary 
        Information Facility (NAIF) (https://naif.jpl.nasa.gov/naif/index.html).
        
        Required JPL SPICE kernels are downloaded as needed and when changed on the web.

        Parameters
        ----------
        spos : npt.ArrayLike (n,3), optional
            An array of spacecraft positions (ECEF - kilometers), 
            by default np.array([6771.0,0,0],ndmin=2)
        stime : npt.ArrayLike (n), optional
            An array of spacecraft times for which the position of body are retrieved.
            The scale of the time is needed in order to derive JPL SPICE ephemeris 
            time (ET). By default pd.Series(pd.to_datetime('2020-01-01')).
        body : npt.ArrayLike, optional
            A list of bodies for which to calculate accelerations at the location 
            of a spacecraft (spos), by default ['SUN'].
        GM : npt.ArrayLike | None, optional
            Mass parameters for the bodies in body. Should be in the same order as body.
            If nothing is passed they are loaded from the constants module which uses
            JPLs de440 mass parameters, by default None.
        scale : str | None, optional
            Time scale of stime; allowed values are 'GPS','TAI','UTC','ET','TDB'.
            By default None but one is required and a Value error will be thrown if it 
            is not passed or if it is not one of the allowed values.
        ephemeris : str, optional
            JPL SPICE ephemeris file to use; allowed values are de440 and de440s. 's'
            denotes the smaller version which covers a shorter time frame. By default 
            'de440s'

        Raises
        ------
        ValueError
            ephemeris not in allowed ephemeris
        ValueError
            scale not in allowed scales
        """    
        # allowed ephemeris to load
        allowed_eph = ['de440','de440s']
        if ephemeris in allowed_eph:
            eph_sh = ephemeris[0:5]

        # download and load the SPICE kernels
        # we want to use
        # checks if the kernels exists or if they've
        # changed online before downloading
        # also checks if they've already been loaded
        # currently loads leap seconds (.tls), ephemeris (.bsp), and 
        # Earth orientation data (.bpc)
        sp_kernels = self.dl_kernels(ephemeris)
        sp_kcnt = spice.ktotal('ALL')
        sp_loaded = [spice.kdata(i,'ALL')[0] for i in range(sp_kcnt)]
        for fp in sp_kernels:
            if fp in sp_loaded:
                print(f'Kernel already loaded - {fp}')
            else:
                print(f'Loading Kernel {fp}')
                spice.furnsh(fp) # need to check if kernels are loaded

        # allowed time scales for getting third body accelerations
        allowed_scales = ['GPS','TAI','UTC','ET','TDB']
        if not scale:
            raise ValueError(f'scale must be one of {allowed_scales}')
        
        scale = scale.upper()
        if scale not in allowed_scales:
            raise ValueError(f'Incorrect scale {scale}, scale must be one of {allowed_scales}')

        # calculate the seconds past j2000 to convert
        # to SPICE ET Ephemeris time (in the SPICE system,
        # this is equivalent to TDB time)
        j2000 = pd.Timestamp('2000-01-01 12:00:00')
        spj2000 = ((pd.Series(stime) - j2000).dt.total_seconds()).to_list()

        # set all needed attributes
        self.et = [spice.unitim(sp_in,scale,'ET') for sp_in in spj2000]
        self.spos = spos
        self.body = [bd.upper() for bd in body]
        self.stime = pd.to_datetime(stime)
        if not GM:
            self.GM = [GMc[eph_sh][bd] for bd in self.body]
        else:
            self.GM = GM

        #attributes used later
        self.bd_ecef = None
        self.bd_acc = None

    def calc_tba(self):
        """Derives third body accelerations from spacecraft positions for solar
        system bodies.

        Uses the SPICE (Spacecraft, Planet, Instrument, C-matrix, Events) observation 
        geometry information system and Kernels from JPLs Navigation and Ancillary 
        Information Facility (NAIF). 
        """
        # get the body positions in ecef
        bd_ecef = np.array([spice.spkpos(bd,self.et,'ITRF93','NONE','EARTH')[0]
                   for bd in self.body])
        
        bd_acc = tba_pairwise_numba(self.spos, bd_ecef, self.GM)

        self.bd_acc = bd_acc
        self.bd_ecef = bd_ecef

    def dl_kernels(self, ephemeris):
        """Download required JPL SPICE kernels.

        Download ephemeris, leap second, and Earth orientation Kernels from JPLs 
        Navigation and Ancillary Information Facility (NAIF). 

        Parameters
        ----------
        ephemeris : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        base_url = 'https://naif.jpl.nasa.gov'
        base_pth = '/pub/naif/generic_kernels'
        
        ephem_d = 'spk/planets'
        leaps_d = 'lsk'
        pck_d = 'pck'

        ephem_f = f'{ephemeris}.bsp'
        leaps_f = 'naif0012.tls'
        pck_f = 'earth_latest_high_prec.bpc'

        ephem_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,ephem_d,ephem_f))
        leaps_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,leaps_d,leaps_f))
        pck_url = urllib.parse.urljoin(base_url,
                                    posixpath.join(base_pth,pck_d,pck_f))

        # find directory of module
        # module directory/swdata/ is where the data is stored
        file_path = Path(__file__).resolve()
        data_path = file_path / '..' / '..' / 'data'
        data_path = data_path.resolve()

        # create it if it doesn't exist
        if not data_path.exists():
            makedirs(data_path)

        for url, file in zip([ephem_url,leaps_url,pck_url],[ephem_f,leaps_f,pck_f]):
            # create file names
            fp = path.join(data_path,file)
            if not path.exists(fp):
                dl_file(url,fp)
            else:
                #check for modification times
                loc_tz = datetime.now().astimezone().tzinfo
                gmt_tz = tz.gettz('GMT')

                mod_file = datetime.fromtimestamp(path.getmtime(fp), tz=loc_tz)
                mod_file = mod_file.astimezone(gmt_tz)
                mod_url = wf_mtime(url)

                if mod_url == None:
                    print(f'Could not determine modification time of {url}')
                elif mod_url > mod_file:
                    print(f'Downloading new version of {url}')
                    dl_file(url,fp)

        return [path.join(data_path,fp) for fp in [ephem_f,leaps_f,pck_f]]

    def get_tba(self):
        """Get the Third Body Accelerations.

        Returns
        -------
        np.Array
            Third body accelerations
        """        
        return self.bd_acc

    def get_body_pos(self):
        """Get the Third Body ECEF Positions.

        Returns
        -------
        np.Array
            Third body positions in ECEF coordinates.
        """
        return self.bd_ecef
