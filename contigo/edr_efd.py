from scipy.integrate import cumulative_simpson
from scipy.integrate import cumulative_trapezoid

import contigo.constants as constants

from .forces.base import ForceModel
from .constellation import Constellation


# ==================================================================
# Energy Dissipation Rate Density (EDRDensity)
# ==================================================================
class EDRDensity:
    """
    Energy Dissipation Rate Density calculator.

    Accepts:
        - Constellation
        - list of ForceModel implementations

    Computes:
        - Total acceleration
        - Conservative potential (if available)
        - Energy dissipation rate (v · a_nonconservative)
    """

    def __init__(self,
                 sc_system: Constellation,
                 force_models: list[ForceModel],
                 potential_model: ForceModel,):
        
        self.system = sc_system
        self.force_models = force_models
        self.potential_model = potential_model

    # --------------------------------------------------------------
    def compute(self) -> dict:
        # compute edr and denomentator
        pass

    def compute_edr(self) -> dict:
        """
        Returns dictionary keyed by spacecraft ID.
        """
        spacecraft_dict = self.system.spacecraft

        results = {}

        earth_angv2 = constants.WGS84_EARTH_ANGULAR_VELOCITY**2 

        for sc_id, sc in spacecraft_dict.items():
            N = sc.N
            
            sc_p = sc.state_ecef[:, 0:3]
            sc_v = sc.state_ecef[:, 3:6]

            sc_xy2 = sc_p[:,0]**2 + sc_p[:,1]**2
            sc_v2 = (sc_v*sc_v).sum(axis=1)

            e_gp = self.potential_model.potential(sc)

            # equation a18 and a19 non integral portion
            # here we use r^2*cos*2(phi) = x^2+y^2
            # and we subtract the edr at edr time zero
            edr = sc_v2/2. - e_gp - earth_angv2*sc_xy2/2.
            edr = edr - edr[0]

            # compute the acceleration integrals
            # x-axis for integrating
            acc_int = np.zeros(N) 
            x_ax = (pd.DatetimeIndex(sc.time).to_julian_date()*86400.).to_numpy()
            x_ax = x_ax-x_ax.min() 
            for model in self.force_models
                acc = model.acceleration(sc)

                a_int = acc*sc_v
                acc_int += cumulative_trapezoid(a_int, x_ax, initial=0)

            edr = edr - acc_int 

            results[sc_id] = {'edr': edr}

        return results
    
    def compute_denom(self) -> dict:
        pass
