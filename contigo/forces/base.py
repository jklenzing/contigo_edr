"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from typing import Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt

from contigo.constellation import Constellation
from contigo.solar_system_ephem import SolarSystemEnvironment

@runtime_checkable
class ForceModel(Protocol):
    """
    All force models operate on a Constellation.

    A single Spacecraft should be represented as a
    Constellation with one member.
    """

    name: str
    is_conservative: bool

    def acceleration(self, 
                     constellation: Constellation,
                     solarsys_env: SolarSystemEnvironment
                     ) -> dict[str, npt.NDArray[np.float64]]:
        ...

    def potential(self, 
                  constellation: Constellation
                  ) -> dict[str, npt.NDArray[np.float64]]:
        ...