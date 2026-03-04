"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 25/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import posixpath
import urllib.parse
import logging

from os import path
from operator import itemgetter
from datetime import datetime, timezone
from dateutil import tz


import numpy as np
import numpy.typing as npt

import spiceypy as spice

import contigo.contig_utils.utils as utils
import contigo.config as config

from contigo.contig_utils.constants import GMc

logger = logging.getLogger(__name__)

class SolarSystemEnvironment:
    """
    High-performance solar system ephemeris cache.

    Design
    ------
    • Bodies are static after initialization
    • Tolerance is static after initialization
    • Internally uses a dictionary keyed by quantized time
    • O(1) lookup instead of O(N²) tolerance scans
    • Cached arrays rebuilt lazily only if needed
    """

    def __init__(
        self,
        bodies: npt.NDArray[np.str_],
        sp_et: npt.NDArray[np.float64] | None,
        sp_gps: npt.NDArray[np.float64] | None,
        tolerance: float | None,
        provider: SPICEEphem,
    ) -> None:

        self.bodies = np.array([b.upper() for b in bodies])
        if tolerance is None:
            self.tolerance = None
        else:
            self.tolerance = float(tolerance)
        self._provider = provider

        # Internal dictionary cache
        # key   -> quantized time (float)
        # value -> (Nb, 3) position array
        self._cache: dict[float, np.ndarray] = {}

        if not isinstance(sp_et, type(None)) and not isinstance(sp_gps, type(None)):
            sp_et = np.asarray(sp_et, dtype=float)
            sp_gps = np.asarray(sp_gps,dtype=float)
            self._load_times(sp_et,sp_gps)

        eph = provider.ephemeris[0:5]

        self.GM = np.array([GMc[eph][bd] for bd in self.bodies],dtype=float)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def get_ephem(
        self,
        sp_et: npt.NDArray[np.float64],
        sp_gps: npt.NDArray[np.float64],
    ) -> tuple[np.ndarray, np.ndarray]:

        sp_et = np.asarray(sp_et, dtype=float)
        sp_gps = np.asarray(sp_gps,dtype=float)

        # Ensure all times are loaded
        self._load_times(sp_et, sp_gps)

        #-----------
        # old method
        #-----------
        # Build output array directly from dictionary
        #nb = len(self.bodies)
        #nt = len(et)
        #r_out = np.empty((nb, nt, 3), dtype=float)

        #for i, t in enumerate(et):
        #    key = self._quantize(t)
        #    r_out[:, i, :] = self._cache[key]

        #-----------
        # new method
        # ~10x faster
        #-----------
        key = self._quantize(sp_gps)
        r_out = np.array(itemgetter(*key)(self._cache))
        r_out = np.swapaxes(r_out,0,1)

        return sp_et, sp_gps, r_out

    # ----------------------------------------------------------
    # Internal
    # ----------------------------------------------------------

    def _quantize(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.int_ | np.float64]:
        """Quantize time using tolerance and return integer bin."""
        if self.tolerance == 0.0:
            # exact integer seconds binning
            return np.round(t).astype(int)
        elif self.tolerance is None:
            return t
        return np.round(t / self.tolerance).astype(int)

    def _load_times(self, 
                    sp_et: np.ndarray,
                    sp_gps: np.ndarray,) -> None:
        """
        Load only times not already cached.
        """

        # Quantize requested times
        q_times = self._quantize(sp_gps)

        # Identify which quantized times are missing
        missing = [[i,t] for i, t in zip(q_times,sp_et) if i not in self._cache]

        if not missing:
            return

        missing_qt, missing_et = zip(*missing)

        # convert the missing qauntized time back 
        # to normal et time
        #missing_et = np.array(missing, dtype=float)
        #if self.tolerance:
        #    missing_et = np.array(missing, dtype=float)*self.tolerance

        # Call provider once for all missing times
        _, r_new = self._provider(self.bodies, missing_et)

        # Store per-time slice in dictionary
        for i, t in enumerate(missing_qt):
            # r_new shape = (Nb, Nt_missing, 3)
            self._cache[t] = r_new[:, i, :]