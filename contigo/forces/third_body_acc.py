"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import posixpath
import urllib.parse
import logging

from os import path
from datetime import datetime, timezone
from dateutil import tz

import pandas as pd
import numpy as np
import numpy.typing as npt
import spiceypy as spice

import contigo.utils as utils
import contigo.config as config

from .tba_utils import tba_pairwise_numba
from ..constants import GMc
from ..constellation import Constellation

logger = logging.getLogger(__name__)

class ThirdBodyAcc:
    """Deriving Third Body Acceleration using JPL SPICE
    """
    def __init__(self, 
                 spos: npt.ArrayLike | None = None,
                 stime: npt.ArrayLike | None = None,
                 body: npt.ArrayLike | None = None,
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
        # set defaults 
        if spos is None:
            spos = np.array([[6771.0, 0.0, 0.0]])
        if stime is None:
            stime = pd.Series(pd.to_datetime("2020-01-01"))
        if body is None:
            body = ["SUN"]

        # allowed ephemeris to load
        allowed_eph = ['de440','de440s']
        if ephemeris in allowed_eph:
            eph_sh = ephemeris[0:5]
        else:
            raise ValueError(f'ephemeris must be on of allowed {allowed_eph}')

        # allowed time scales for getting third body accelerations
        allowed_scales = ['GPS','TAI','UTC','ET','TDB']
        if not scale:
            raise ValueError(f'scale must be one of {allowed_scales}')
        
        scale = scale.upper()
        if scale not in allowed_scales:
            raise ValueError(f'Incorrect scale {scale}, scale must be one of {allowed_scales}')

        self.spos = np.asarray(spos, dtype=float)
        if self.spos.ndim != 2 or self.spos.shape[1] != 3:
            raise ValueError("spos must have shape (N,3)")

        self.stime = pd.to_datetime(stime)
        if self.stime.shape[0] != self.spos.shape[0]:
            raise ValueError("spos and stime must have be (N,3) and (N)")

        self.body = [bd.upper() for bd in body]
        
        self.scale = scale
        if GM is None:
            self.GM = np.array([GMc[eph_sh][bd] for bd in self.body],dtype=float)
        else:
            self.GM = np.asarray(GM, dtype=float)

        if len(self.body) != len(self.GM):
            raise ValueError("body and GM must be same length")
        self.ephemeris = ephemeris
        #attributes used later
        self.bd_ecef = None
        self.bd_acc = None

        self.load_kernels()

    def calc_tba(self):
        """Derives third body accelerations from spacecraft positions for solar
        system bodies.

        Uses the SPICE (Spacecraft, Planet, Instrument, C-matrix, Events) observation 
        geometry information system and Kernels from JPLs Navigation and Ancillary 
        Information Facility (NAIF).
        """
        # calculate the seconds past j2000 to convert
        # to SPICE ET Ephemeris time (in the SPICE system,
        # this is equivalent to TDB time)
        if self.stime.dt.tz is not None and self.stime.dt.tz != timezone.utc:
            print(str(self.stime.dt.tz))
            raise ValueError('stime should be time zone naive or UTC')
        j2000 = pd.Timestamp('2000-01-01 12:00:00')
        spj2000 = ((self.stime - j2000).dt.total_seconds()).to_list()

        # set all needed attributes
        et = [spice.unitim(sp_in,self.scale,'ET') for sp_in in spj2000]

        # get the body positions in ecef
        bd_ecef = np.array([spice.spkpos(bd,et,'ITRF93','NONE','EARTH')[0]
                   for bd in self.body])

        bd_acc = tba_pairwise_numba(self.spos, bd_ecef, self.GM)

        self.bd_acc = bd_acc
        self.bd_ecef = bd_ecef

    def load_kernels(self):
        """Download and load the SPICE kernels for deriving body locations

        Checks if an attemp at downloading the kernels has already been done.

        Download will check if files have changed and need to be redownloaded.
        Checks if kernels are already loaded.

        Loads the kernels.

        Currently loads leap seconds (.tls), ephemeris (.bsp), and Earth orientation 
        data (.bpc)
        """
        # set file names 
        ephem_f = f'{self.ephemeris}.bsp'
        leaps_f = 'naif0012.tls'
        pck_f = 'earth_latest_high_prec.bpc'
        # get file paths
        sp_kernels = [path.join(config.DATA_DIR,fp) for fp in [ephem_f,leaps_f,pck_f]]

        # check if we've already attempted to download kernels
        if config.state['kernel_downloaded'] is False:
            self.dl_kernels(ephem_f, leaps_f, pck_f)

        # check if kernels are already loaded, load them if not
        sp_kcnt = spice.ktotal('ALL')
        sp_loaded = [spice.kdata(i,'ALL')[0] for i in range(sp_kcnt)]
        for fp in sp_kernels:
            if fp in sp_loaded:
                logger.info('Kernel already loaded - %s', fp)
            else:
                logger.info('Loading Kernel - %s', fp)
                spice.furnsh(fp) # need to check if kernels are loaded

    def dl_kernels(self, ephem_f: str, leaps_f: str, pck_f:str):
        """Download required JPL SPICE kernels.

        Download ephemeris, leap second, and Earth orientation Kernels from JPLs 
        Navigation and Ancillary Information Facility (NAIF). 

        Will check if local files exist. Will download if files on the server are newer
        then the local files.

        Parameters
        ----------
        ephem_f :str
            JPL ephemeris file to download.
        leaps_f : str
            JPL leap seconds file to download.
        pck_f : str
            JPL Earth orientation file to download.
        """
        base_url = 'https://naif.jpl.nasa.gov'
        base_pth = '/pub/naif/generic_kernels'
        
        ephem_d = 'spk/planets'
        leaps_d = 'lsk'
        pck_d = 'pck'

        ephem_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,ephem_d,ephem_f))
        leaps_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,leaps_d,leaps_f))
        pck_url = urllib.parse.urljoin(base_url,
                                    posixpath.join(base_pth,pck_d,pck_f))

        for url, file in zip([ephem_url,leaps_url,pck_url],[ephem_f,leaps_f,pck_f]):
            # create file names
            fp = path.join(config.DATA_DIR,file)
            if not path.exists(fp):
                logger.info('Downloading kernel - %s', fp)
                utils.dl_file(url,fp)
            else:
                #check for modification times
                loc_tz = datetime.now().astimezone().tzinfo
                gmt_tz = tz.gettz('GMT')

                mod_file = datetime.fromtimestamp(path.getmtime(fp), tz=loc_tz)
                mod_file = mod_file.astimezone(gmt_tz)
                mod_url = utils.wf_mtime(url)

                if mod_url == None:
                    logger.info('Could not determine modification time of %s', url)
                elif mod_url > mod_file:
                    logger.info('Downloading new version of %s', url)
                    utils.dl_file(url,fp)
        
        config.state['kernel_downloaded'] = True

    def get_tba(self):
        """Get the Third Body Accelerations.

        Returns
        -------
        np.Array
            Third body accelerations
        """
        if self.bd_acc is None:
            raise RuntimeError("calc_tba() must be called first")
        return self.bd_acc

    def get_body_pos(self):
        """Get the Third Body ECEF Positions.

        Returns
        -------
        np.Array
            Third body positions in ECEF coordinates.
        """
        if self.bd_ecef is None:
            raise RuntimeError("calc_tba() must be called first")
        return self.bd_ecef


class ThirdBody:
    """
    Third-body gravity force operating on invdividual satellites in a Constellation
    object.
    """

    name: str = "ThirdBodyAcceleration"

    def __init__(self,
                 body=None,
                 GM=None,
                 ephemeris: str = "de440s",):
        """Third body gravity acting on individual satellites in a Constellation
        
        Wrapper for ThirdBodyAcc to follow the .base.ForceModel(Protocol)

        Parameters
        ----------
        body : npt.ArrayLike, optional
            A list of bodies for which to calculate accelerations at the location 
            of a spacecraft (spos), by default ['SUN'].
        GM : npt.ArrayLike | None, optional
            Mass parameters for the bodies in body. Should be in the same order as body.
            If nothing is passed they are loaded from the constants module which uses
            JPLs de440 mass parameters, by default None.
        ephemeris : str, optional
            JPL SPICE ephemeris file to use; allowed values are de440 and de440s. 's'
            denotes the smaller version which covers a shorter time frame. By default 
            'de440s'

        """        
        self.body = body
        self.GM = GM
        self.ephemeris = ephemeris

    def acceleration(self, 
                     constellation: Constellation
                     ) -> dict[str, npt.NDArray[np.float64]]:
        """Derive third body accellerations.

        Use ThirdBodyAcc to derive accelerations for satellites in a Constellation 
        object.

        Constellation holds the state and time scale of all satellites.

        Initialization defines the ephemeris to use and the solar system bodies

        Returns:
            dict[spacecraft_id] -> (N,3)
        """

        acc_dict = {}

        for sc_id, sc in constellation.spacecraft.items():

            tba = ThirdBodyAcc(
                spos=sc.state_ecef[:, 0:3],
                stime=sc.stime,
                body=self.body,
                GM=self.GM,
                scale=sc.tscale,
                ephemeris=self.ephemeris,
            )

            tba.calc_tba()
            acc_dict[sc_id] = tba.get_tba()

        return acc_dict

    def potential(self, 
                  constellation: Constellation
                  ) -> dict[str, npt.NDArray[np.float64]]:
        raise NotImplementedError("Not implemented for ThirdBodyAcc.")