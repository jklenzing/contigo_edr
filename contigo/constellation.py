"""Derive third body accelerations for an Earth orbiting spacecraft.

added: 18/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

import numpy as np

from .spacecraft import Spacecraft

# ==================================================================
# Constellation Container
# ==================================================================
class Constellation:
    """
    Constellation container that accepts the SAME inputs as Spacecraft,
    performs loading once, and then splits into individual Spacecraft
    objects using Spacecraft.split_by_id().

    This preserves all loading logic while enforcing that each member
    spacecraft is a strict single-ID Spacecraft object.
    """

    def __init__(self, **spacecraft_kwargs):
        # Load once using the Spacecraft loader
        multi_sc = Spacecraft(**spacecraft_kwargs)

        # Split into individual spacecraft
        self.spacecraft: dict = multi_sc.split_by_id()
        # Get constellation times from spacecraft
        # keep only unique times
        self.sspice_gps, u_id = np.unique(multi_sc.sspice_gps, return_index=True)
        self.sspice_et = multi_sc.sspice_et[u_id]
        self.sc_utc = multi_sc.sc_utc[u_id]

    @classmethod
    def _from_add(cls, spacecraft_dict, sspice_et, sspice_gps, sc_utc):
        obj = cls.__new__(cls)
        obj.spacecraft = spacecraft_dict
        obj.sspice_et = sspice_et
        obj.sspice_gps = sspice_gps
        obj.sc_utc = sc_utc
        return obj

    # --------------------------------------------------------------
    @property
    def ids(self) -> list:
        return list(self.spacecraft.keys())

    def __getitem__(self, key):
        return self.spacecraft[key]

    def __iter__(self):
        return iter(self.spacecraft.values())

    def __len__(self):
        return len(self.spacecraft)

    def __repr__(self) -> str:
        return f"Constellation(n_spacecraft={len(self)}, ids={self.ids})"
    
    def __add__(self, other: "Constellation") -> "Constellation":
        if not isinstance(other, Constellation):
            return NotImplemented
        
        # Prevent duplicate spacecraft IDs
        overlap = set(self.ids).intersection(other.ids)
        if overlap:
            raise ValueError(
                f"Cannot merge constellations. Duplicate spacecraft IDs found: {overlap}"
            )
        
        merged_dict = self.spacecraft | other.spacecraft
        merged_et = self._merge_time_array(self.sspice_et, other.sspice_et)
        merged_gps = self._merge_time_array(self.sspice_gps, other.sspice_gps)
        merged_utc = self._merge_time_array(self.sc_utc, other.sc_utc)

        return Constellation._from_add(
            spacecraft_dict=merged_dict,
            sspice_et=merged_et,
            sspice_gps=merged_gps,
            sc_utc=merged_utc)
    
    def __iadd__(self, other: "Constellation") -> "Constellation":
        if not isinstance(other, Constellation):
            return NotImplemented

        overlap = set(self.ids).intersection(other.ids)
        if overlap:
            raise ValueError(
                f"Cannot merge constellations. Duplicate spacecraft IDs found: {overlap}"
            )

        self.spacecraft.update(other.spacecraft)

        self.sspice_et = self._merge_time_array(self.sspice_et, other.sspice_et)
        self.sspice_gps = self._merge_time_array(self.sspice_gps, other.sspice_gps)
        self.sc_utc = self._merge_time_array(self.sc_utc, other.sc_utc)

        return self

    @staticmethod
    def _merge_time_array(a, b):
        """
        Merge two time arrays safely:
        - Handles numpy arrays or lists
        - Removes duplicates
        - Returns sorted unique array
        """
        if a is None:
            return b
        if b is None:
            return a

        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        merged = np.concatenate((a_arr, b_arr))
        return np.unique(merged)
