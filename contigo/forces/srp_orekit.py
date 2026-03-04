import numpy as np
import numpy.typing as npt

from contigo.forces.base import ForceModel
from contigo.constellation import Constellation
from contigo.forces.srp_gmat import SRPGMATAcc
from contigo.solar_system_ephem import SolarSystemEnvironment

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.contigo.orekit_utils import SRPCannonballBatchHelper

class SRPOrekitCB(ForceModel):
    """
    SRP accelerations for invdividual satellites in a Constellation object.
    """

    name: str = "SRPOrekitCannonball"

    def __init__(self):
        """SRP acceleration for individual satellets in a Constellation object.

        Wrapper for the batch processing java class SRPCannonballBatchHelper.
         
        Requires Orekit and the JVM to be running.

        Parameters
        ----------
        """        
        
        # we eventually want to check that the orekit JVM is running and 
        # that the SRPCannonballBatchHelper class is available, 
        # but for now we will just assume it is and catch errors when we try to use it.
        
        pass

    def acceleration(self, 
                     constellation: Constellation,
                     solarsys_env: SolarSystemEnvironment
                     ) -> dict[str, npt.NDArray[np.float64]]:
        """Derive SRP accelerations. 

        Use SRPCannonballBatchHelper to derive *cannonball* SRP accelerations 
        for spacecraft in a Constellation object.

        Constellation holds the state and time for all satellites. 

        Parameters
        ----------
        constellation : Constellation
            Constellation container of Spacecraft objects.

        Returns
        -------
        dict[spacecraft_id] -> (N,3)
        """        
        acc_dict = {}
        utc = TimeScalesFactory.getUTC()
        
        for sc_id, sc in constellation.spacecraft.items():

            utc_time = sc.sc_utc
            first_dt = utc_time[0]
            ref_date = AbsoluteDate(first_dt.year, first_dt.month, first_dt.day,
                                        first_dt.hour, first_dt.minute,
                                        float(first_dt.second), utc) 
            offsets = np.array(
                 [(t - first_dt).total_seconds() for t in utc_time],
                 dtype=np.float64)
            
            acc = SRPCannonballBatchHelper.get_acc(ref_date, offsets,
                                                    sc.state*1000.,
                                                    sc.sc_mass_arr,
                                                    sc.srp_area_arr,
                                                    sc.cr_arr)
            acc = np.array(acc)
            # get only the ecef acceleration and convert back to km/s^2
            acc_dict[sc_id] = acc[:,3:]/1000.

        return acc_dict

    def potential(self, 
                constellation: Constellation
                ) -> dict[str, npt.NDArray[np.float64]]:
        raise NotImplementedError("Not implemented for SRPAcc.")