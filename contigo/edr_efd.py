"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 23/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import pandas as pd
import numpy as np

from scipy.integrate import cumulative_simpson

import contigo.contigo_utils.constants as constants

from contigo.forces.base import ForceModel
from contigo.constellation import Constellation
from contigo.solar_system_ephem import SolarSystemEnvironment


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
                 constellation: Constellation,
                 solarsys_env: SolarSystemEnvironment,
                 force_models: list[ForceModel],
                 potential_model: ForceModel,):
        
        self.constellation = constellation
        self.solarsys_env = solarsys_env 
        self.force_models = force_models
        self.potential_model = potential_model

        unique_gps, u_id = np.unique(constellation.sspice_gps, return_index=True)

        # load ephemeris for all unique times
        self.solarsys_env._load_times(ephem_time=constellation.sspice_et[u_id], 
                                      gps_time=unique_gps, 
                                      utc_time=constellation.sc_utc)

    # --------------------------------------------------------------
    def compute(self) -> dict:
        # compute edr and denomentator
        pass

    def compute_edr(self) -> dict:
        """
        Compute EDR (Delta Epsilon) from equation A18 and A19 of
        https://doi.org/10.1029/2024EA003898

        Returns dictionary keyed by spacecraft ID.
        """
        # system is the constellation of spacecraft
        spacecraft_dict = self.constellation.spacecraft

        results = {}

        earth_angv2 = constants.WGS84_EARTH_ANGULAR_VELOCITY**2

        # derive the earth potential for each sc in the 
        # constellation
        e_gp_con = self.potential_model.potential(self.constellation)

        self.e_gp = e_gp_con

        # derive the accelerations from the force models
        # for the constellation
        acc_con = { }
        for model in self.force_models:
            acc_con[model.name] = model.acceleration(self.constellation, self.solarsys_env)

        self.accelerations = acc_con

        for sc_id, sc in spacecraft_dict.items():
            N = sc.N
            
            sc_p = sc.state_ecef[:, 0:3]
            sc_v = sc.state_ecef[:, 3:]

            sc_xy2 = sc_p[:,0]**2 + sc_p[:,1]**2
            sc_v2 = (sc_v*sc_v).sum(axis=1)

            e_gp = e_gp_con[sc_id]

            # equation a18 and a19 non integral portion
            # here we use r^2*cos*2(phi) = x^2+y^2
            # and we subtract the edr at edr time zero
            edr = sc_v2/2. - e_gp - earth_angv2*sc_xy2/2.
            #edr = sc_v2/2. - earth_angv2*sc_xy2/2.

            # compute the acceleration integrals
            # x-axis for integrating
            acc_int = np.zeros(N)
            x_ax = sc.sspice_gps
            x_ax = x_ax-x_ax.min()
            for m_id, m_acc in acc_con.items():
                # if the force model returns multiple accelerations
                # loop through them all
                print(m_id)
                if m_acc[sc_id].shape[0] != N:
                    for i in range(m_acc[sc_id].shape[0]):
                        acc = m_acc[sc_id][i,:,:]
                        if acc.shape[0] != N:
                            raise ValueError('Model accelerations should be shape (N,3) or (x,N,3)')
                        a_int = (acc*sc_v).sum(axis=1)
                        acc_int += cumulative_simpson(a_int, x=x_ax, initial=0)
                else:
                    a_int = (m_acc[sc_id]*sc_v).sum(axis=1)
                    acc_int += cumulative_simpson(a_int, x=x_ax, initial=0)

            edr = edr - acc_int
            edr = edr - edr[0] 

            results[sc_id] = {'edr': edr}

        return results
    
    def compute_denom(self) -> dict:
        """
        Compute denomenator of equation A20 of
        https://doi.org/10.1029/2024EA003898

        Returns dictionary keyed by spacecraft ID.
        """
        # system is the constellation of spacecraft
        spacecraft_dict = self.constellation.spacecraft

        denom = {}

        for sc_id, sc in spacecraft_dict.items():
            
            b=sc.cd_arr*(sc.drag_area_arr/1000.**2)/sc.sc_mass_arr
            
            sc_v = sc.state_ecef[:, 3:]
            sc_v3 = np.linalg.norm(sc_v,axis=1)**3

            x_ax = sc.sspice_gps
            x_ax = x_ax-x_ax.min()

            y_int = b*sc_v3

            denom[sc_id] = cumulative_simpson(y_int,x=x_ax,initial=0)

        return denom