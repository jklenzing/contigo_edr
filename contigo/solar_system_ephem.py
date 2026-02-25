
import posixpath
import urllib.parse
import logging

from os import path
from datetime import datetime, timezone
from dateutil import tz

import numpy as np
import numpy.typing as npt

import spiceypy as spice

import contigo.utils as utils
import contigo.config as config

logger = logging.getLogger(__name__)

# ==============================================================
# Ephemeris Provider
# ============================================================== 

class SPICE_Ephem:
    """
    SPICE-backed ephemeris provider with unique-time optimization.
    """

    def __init__(self, 
                 ephemeris: str='de440s',
                 frame: str = "ITRF93", 
                 observer: str = "EARTH"):
        self.frame = frame
        self.observer = observer
        self.ephemeris= ephemeris

        self.load_kernels()

    def __call__(self, body: npt.NDArray[np.str_], et: np.ndarray):
        
        # find the unique values of et and return their indecies
        # return the inverse indices of that allow the reconstruction
        # of the orignal array
        unique_et, inv = np.unique(et, return_inverse=True)

        r_unique = np.array(
            [
            spice.spkpos(bd.upper(), unique_et, self.frame,'NONE',self.observer)[0]
            for bd in body
            ])

        r_body = r_unique[:,inv,:]
        return et, r_body
    
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