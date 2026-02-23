"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 23/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
from pathlib import Path
import os


import numpy as np
import numpy.typing as npt
import pandas as pd

import contigo.config as config

from contigo.forces import srp_utils

class SRPGMATAcc:


    def __init__(self,
                 sc_state: npt.ArrayLike | None = None,
                 sc_time: npt.ArrayLike | None = None,
                 sc_cr: npt.ArrayLike | None = None,
                 sc_srparea: npt.ArrayLike | None = None,
                 sc_mass:  npt.ArrayLike | None = None,
                 apistartup: str | Path, 
                 gmat_install: str | Path):
        
        #first need to setup and load GMAT Python API
        srp_utils.setup_gmat(apistartup,gmat_install)

        self.state = sc_state
        self.time = sc_time
        self.cr = sc_cr
        self.srp_area = sc_srparea
        self.mass = sc_mass

    def calc_srp(self):

        gmat = config.state['gmatpy']

        gtime = pd.to_datetime(self.stime,format='%d %b %Y %H:%M:%S.000')
        #setup a gmat spacecraft
        earthorb = gmat.Construct("Spacecraft", "EarthOrbiter")
        earthorb.SetField("DateFormat", "TAIGregorian")
        earthorb.SetField("Epoch", f'{gtime[0]}')
        # Spacecraft State
        earthorb.SetField("CoordinateSystem", "EarthFixed")
        earthorb.SetField("X", x)
        earthorb.SetField("Y", y)
        earthorb.SetField("Z", z)
        earthorb.SetField("VX", vx)
        earthorb.SetField("VY", vy)
        earthorb.SetField("VZ", vz)
        earthorb.SetField("DryMass", self.mass[0])
        earthorb.SetField("Cr", self.cr)
        earthorb.SetField("SRPArea", self.cr_area)

        #setup forces to get SRP
        fm = gmat.Construct("ForceModel", "FM")
        fm.SetField("CentralBody", "Earth")

        # Solar Radiation Pressure
        srp = gmat.Construct("SolarRadiationPressure", "SRP")
        fm.AddForce(srp)

        # Setup the state vector used for the force
        psm = gmat.PropagationStateManager()
        psm.SetObject(earthorb)
        psm.BuildState()

        fm.SetPropStateManager(psm)
        fm.SetState(psm.GetState())

        # Assemble all of the objects together 
        gmat.Initialize()

        # Finish force model setup:
        ##  Map spacecraft state into the model
        fm.BuildModelFromMap()
        ##  Load physical parameters needed for the forces
        fm.UpdateInitialData()

        # Create the converter
        csConverter = gmat.CoordinateConverter()

        # Create the input and output coordinate systems
        eci  = gmat.Construct("CoordinateSystem", "ECI", "Earth", "MJ2000Eq")
        ecef = gmat.Construct("CoordinateSystem", "ECEF", "Earth", "BodyFixed")
        gmat.Initialize()

        srp_der = []
        srp_der_ecef = []

        for st, ep, cr, srp_a, mass in zip(self.state,gtime,self.cr, self.srp_area, self.mass):
            earthorb.SetField("Epoch", f"{ep}")
            earthorb.SetField("X", st[0])
            earthorb.SetField("Y", st[1])
            earthorb.SetField("Z", st[2])
            earthorb.SetField("VX", st[3])
            earthorb.SetField("VY", st[4])
            earthorb.SetField("VZ", st[5])

            earthorb.SetField("Cr", cr)
            earthorb.SetField("SRPArea", srp_a)

            # Solar Radiation Pressure
            srp = gmat.Construct("SolarRadiationPressure", "SRP")

            # Add all of the forces into the ODEModel container
            # ODE Model settings
            fm = gmat.Construct("ForceModel", "FM")
            fm.SetField("CentralBody", "Earth")
            fm.AddForce(srp)

            # Setup the state vector used for the force
            psm = gmat.PropagationStateManager()
            psm.SetObject(earthorb)
            psm.BuildState()

            fm.SetPropStateManager(psm)
            fm.SetState(psm.GetState())

            gmat.Initialize()

            # Finish force model setup:
            ##  Map spacecraft state into the model
            fm.BuildModelFromMap()
            ##  Load physical parameters needed for the forces
            fm.UpdateInitialData()

            pstate = earthorb.GetState().GetState()

            srp.GetDerivatives(pstate)
            srp_der.append(srp.GetDerivativeArray())

            # rotate the accelerations into ecef
            conv_vec = gmat.Rvector6(0,0,0,0,0,0)

            # SRP
            csConverter.Convert(earthorb.GetEpoch(), gmat.Rvector6(srp_der[-1]), eci, conv_vec, ecef)
            srp_der_ecef.append(conv_vec.GetDataVector())







    



