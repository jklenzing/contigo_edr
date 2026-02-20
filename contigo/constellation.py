
from spacecraft import Spacecraft

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

        # Preserve shared time scale
        self.tscale = multi_sc.tscale

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