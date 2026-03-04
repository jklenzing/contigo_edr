
import numpy as np
import numpy.typing as npt

from java.util import ArrayList

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.contigo.orekit_utils import EphemerisBatchHelper

class OrekitEphem:
    """
    Orekit-backed ephemeris provider.
    """

    def __init__(self, 
                 ephemeris: str='de440s'):
        
        self.ephemeris= ephemeris

    def __call__(self, body: npt.NDArray[np.str_] | list[str],
                 utc_time: np.ndarray | None = None,
                 gps_time: None = None,
                 ephem_time: None = None,
                 ):
        
        utc = TimeScalesFactory.getUTC()
        first_dt = utc_time[0]
        ref_date = AbsoluteDate(first_dt.year, first_dt.month, first_dt.day,
                                        first_dt.hour, first_dt.minute,
                                        float(first_dt.second), utc) 
        offsets = np.array(
            [(t - first_dt).total_seconds() for t in utc_time],
            dtype=np.float64)

        r_body = EphemerisBatchHelper.getBodyECEF(ref_date, 
                                                  offsets, 
                                                  ArrayList(['Sun', 'Moon']))
        
        # convert to km and return
        return np.array(r_body, dtype=np.float64)/1000.0