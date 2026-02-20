from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import os
import numpy as np
import numpy.typing as npt
import pandas as pd
import glob

##TODO Future proof units

@dataclass
class Spacecraft:
    """
    Spacecraft state container with explicit separation between
    raw user inputs and internal normalized state variables.

    Raw inputs may be provided directly (state, time) OR loaded
    from one or more files on disk. Internal state is always strict
    and guaranteed after loading.

    State is assumed to be ECEF:
    - position  [x, y, z]
    - velocity  [vx, vy, vz]
    """

    # ------------------------------------------------------------------
    # Raw user inputs (flexible)
    # ------------------------------------------------------------------
    state: npt.ArrayLike | None = None        # (N,6) [x,y,z,vx,vy,vz]
    time: npt.ArrayLike | None = None         # (N,)
    sc_id_input: npt.ArrayLike | None = None  # (N,)
    tscale_input: str | None = None          # ['GPS','TAI','UTC','ET','TDB']
    unit_input: str = 'km'

    # Spacecraft physical properties (scalar or (N,))
    cd: float | npt.ArrayLike | None = None
    drag_area: float | npt.ArrayLike | None = None
    sc_mass: float | npt.ArrayLike | None = None
    cr: float | npt.ArrayLike | None = None
    srp_area: float | npt.ArrayLike | None = None

    # Optional file input
    state_file: str | Path | Iterable[str | Path] | None = None

    # ------------------------------------------------------------------
    # Loader configuration (schema mapping)
    # ------------------------------------------------------------------
    time_col: str = "time"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"
    vx_col: str = "vx"
    vy_col: str = "vy"
    vz_col: str = "vz"
    sc_id_col: str | None = None
    sc_fn_slc: slice | None = None

    # Optional physical-property columns
    cd_col: str | None = None
    drag_area_col: str | None = None
    sc_mass_col: str | None = None
    cr_col: str | None = None
    srp_area_col: str | None = None

    # ------------------------------------------------------------------
    # Internal normalized state (strict)
    # ------------------------------------------------------------------
    state_ecef: npt.NDArray[np.float64] = field(init=False)  # (N,6)
    stime: pd.DatetimeIndex = field(init=False)              # (N,)
    tscale: str = field(init=False)                          # validated time scale
    sc_id: npt.NDArray = field(init=False)                   # (N,)
    unique_ids: npt.NDArray = field(init=False)

    # Normalized spacecraft properties (always (N,))
    cd_arr: npt.NDArray = field(init=False)
    drag_area_arr: npt.NDArray = field(init=False)
    sc_mass_arr: npt.NDArray = field(init=False)
    cr_arr: npt.NDArray = field(init=False)
    srp_area_arr: npt.NDArray = field(init=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self):
        self._validate_timescale()
        self.unit = self.unit_input.lower()

        if self.state is not None or self.time is not None:
            if self.state is None or self.time is None:
                raise ValueError("Both state and time must be provided together")
            self.load_from_arrays(self.state, self.time, self.sc_id_input)
        else:
            if self.state_file is None:
                raise ValueError("Either (state, time) or state_file must be provided")
            self.load_from_file(self.state_file)
        
        # normalize units
        self._normalize_units()

    # ------------------------------------------------------------------
    # Time scale validation
    # ------------------------------------------------------------------
    def _validate_timescale(self) -> None:
        allowed = {'GPS', 'TAI', 'UTC', 'ET', 'TDB'}
        if self.tscale_input is None:
            self.tscale = 'UTC'
        else:
            tscale = self.tscale_input.upper()
            if tscale not in allowed:
                raise ValueError(f"tscale_input must be one of {allowed}")
            self.tscale = tscale

    # ------------------------------------------------------------------
    # Loader / normalizer (array-based)
    # ------------------------------------------------------------------
    def load_from_arrays(
        self,
        state: npt.ArrayLike,
        time: npt.ArrayLike,
        sc_id: npt.ArrayLike | None = None,
    ) -> None:
        """Load spacecraft state from in-memory arrays."""

        s = np.asarray(state, dtype=float)
        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError("state must have shape (N,6)")

        t = pd.to_datetime(time, utc=True)
        if len(t) != s.shape[0]:
            raise ValueError("state and time must have the same length")

        # Spacecraft ID handling
        if sc_id is not None:
            sc = np.asarray(sc_id)
            if sc.ndim != 1 or len(sc) != len(t):
                raise ValueError("sc_id must be 1D and same length as time")
            sc_id_arr = pd.Series(sc).astype("category").to_numpy()
        else:
            sc_id_arr = pd.Series(
                np.full(len(t), "NO_ID", dtype=object)
            ).astype("category").to_numpy()

        # Commit normalized state
        self.state_ecef = s
        self.stime = t
        self.sc_id = sc_id_arr
        self.unique_ids = np.unique(self.sc_id)

        # Normalize physical properties
        self._normalize_properties(len(t), df=None)

    # ------------------------------------------------------------------
    # Loader / normalizer (file-based)
    # ------------------------------------------------------------------
    def load_from_file(
        self,
        state_file: str | Path | Iterable[str | Path],
        loader: str | None = None,
        read_kwargs: dict | None = None,
    ) -> None:
        """Load spacecraft state from one or more files."""

        files = self._expand_files(state_file)
        frames: list[pd.DataFrame] = []

        for file in files:
            df = self._load_table(file, loader, read_kwargs)
            if self.sc_id_col is not None and self.sc_id_col.lower() == 'filename':
                base_file = os.path.basename(file)
                if isinstance(self.sc_fn_slc, slice):
                    base_file = base_file[self.sc_fn_slc]
                df['filename'] = base_file
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)

        required = {
            self.time_col,
            self.x_col, self.y_col, self.z_col,
            self.vx_col, self.vy_col, self.vz_col,
        }

        missing = required - set(df_all.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Normalize time
        t = pd.to_datetime(df_all[self.time_col], utc=True)

        # Normalize state vector
        s = df_all[[
            self.x_col, self.y_col, self.z_col,
            self.vx_col, self.vy_col, self.vz_col,
        ]].to_numpy(dtype=float)

        if s.ndim != 2 or s.shape[1] != 6:
            raise ValueError("State data must have shape (N,6)")
        if len(t) != s.shape[0]:
            raise ValueError("Time and state must have equal length")

        # Spacecraft ID handling
        if self.sc_id_col is not None and self.sc_id_col in df_all.columns:
            sc_id_arr = df_all[self.sc_id_col].astype("category").to_numpy()
        else:
            sc_id_arr = pd.Series(
                np.full(len(t), "NO_ID", dtype=object)
            ).astype("category").to_numpy()

        # Commit normalized state
        self.state_ecef = s
        self.stime = t
        self.sc_id = sc_id_arr
        self.unique_ids = np.unique(self.sc_id)

        # Normalize physical properties (from columns if available)
        self._normalize_properties(len(t), df=df_all)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_properties(self, N: int, df: pd.DataFrame | None = None) -> None:
        """Normalize scalar or array spacecraft properties to strict (N,) arrays."""

        def norm(val, col):
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim == 0:
                    return np.full(N, float(arr))
                if arr.ndim == 1 and len(arr) == N:
                    return arr.astype(float)
                raise ValueError("Property must be scalar or length N")
            if df is not None and col is not None and col in df.columns:
                return df[col].to_numpy(dtype=float)
            return np.full(N, np.nan)

        self.cd_arr = norm(self.cd, self.cd_col)
        self.drag_area_arr = norm(self.drag_area, self.drag_area_col)
        self.sc_mass_arr = norm(self.sc_mass, self.sc_mass_col)
        self.cr_arr = norm(self.cr, self.cr_col)
        self.srp_area_arr = norm(self.srp_area, self.srp_area_col)

    def _normalize_units(self):
        if self.unit in ["m", "meter", "metre"]:
            self.state_ecef /= 1000.0
        elif self.unit in ["km", "kilometer", "kilometre"]:
            pass
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")
        
    # ------------------------------------------------------------------
    def _expand_files(
        self,
        state_file: str | Path | Iterable[str | Path],
    ) -> list[Path]:
        """Expand wildcards and iterables into a sorted list of files."""
        if isinstance(state_file, (str, Path)):
            paths = glob.glob(str(state_file))
        else:
            paths = []
            for f in state_file:
                paths.extend(glob.glob(str(f)))

        files = [Path(p) for p in paths]
        if not files:
            raise FileNotFoundError("No matching state files found")

        return sorted(files)

    def _load_table(
        self,
        file: Path,
        loader: str | None = None,
        read_kwargs: dict | None = None,
    ) -> pd.DataFrame:
        """Dispatch to the appropriate reader based on loader or file extension."""
        suffix = file.suffix.lower()
        loader = loader or suffix.lstrip(".")
        kwargs = read_kwargs or {}

        base_txt = {"csv", "txt", "text"}
        zip_txt = {"gz", "bz2", "zip", "xz", "zst",
                   "tar", "tar.gz", "tar.xz", "tar.bz2"}

        if loader in (base_txt | zip_txt):
            return pd.read_csv(file, **kwargs)
        elif loader in {"hdf", "h5"}:
            return pd.read_hdf(file, **kwargs)
        elif loader == "spk":
            raise NotImplementedError("SPK loader not yet implemented")
        else:
            raise ValueError(f"Unsupported loader type: {loader}")

    # ------------------------------------------------------------------
    # Multi-spacecraft utilities
    # ------------------------------------------------------------------
    def split_by_id(self) -> dict:
        """
        Split a multi-ID Spacecraft container into individual
        Spacecraft objects keyed by spacecraft ID.

        Returns
        -------
        dict
            {spacecraft_id: Spacecraft}
        """
        spacecraft_dict = {}

        for uid in self.unique_ids:
            mask = self.sc_id == uid

            sc = Spacecraft(
                state=self.state_ecef[mask],
                time=self.stime[mask],
                sc_id_input=np.full(mask.sum(), uid),
                tscale_input=self.tscale,
                cd=self.cd_arr[mask],
                drag_area=self.drag_area_arr[mask],
                sc_mass=self.sc_mass_arr[mask],
                cr=self.cr_arr[mask],
                srp_area=self.srp_area_arr[mask],
            )

            spacecraft_dict[uid] = sc

        return spacecraft_dict

    # ------------------------------------------------------------------
    # Grouped normalized state view this might create overhead if we are
    # grabbing data lots but it cleans up the namespace.
    # If it proves to be a problem then we might have to refactor this
    # so that this is done above.
    # ------------------------------------------------------------------
    @dataclass
    class SpacecraftState:
        state_ecef: npt.NDArray[np.float64]
        stime: pd.DatetimeIndex
        sc_id: npt.NDArray
        unique_ids: npt.NDArray
        cd: npt.NDArray
        drag_area: npt.NDArray
        sc_mass: npt.NDArray
        cr: npt.NDArray
        srp_area: npt.NDArray

        @property
        def N(self) -> int:
            return self.state_ecef.shape[0]

        @property
        def r(self) -> npt.NDArray[np.float64]:
            return self.state_ecef[:, 0:3]

        @property
        def v(self) -> npt.NDArray[np.float64]:
            return self.state_ecef[:, 3:6]

    @property
    def state_data(self) -> "Spacecraft.SpacecraftState":
        """
        Structured grouped view of canonical internal state.

        Cached after first access.
        Cache automatically invalidated whenever state is reloaded.
        """
        if not hasattr(self, "_state_data_cache"):
            self._state_data_cache = Spacecraft.SpacecraftState(
                state_ecef=self.state_ecef,
                stime=self.stime,
                sc_id=self.sc_id,
                unique_ids=self.unique_ids,
                cd=self.cd_arr,
                drag_area=self.drag_area_arr,
                sc_mass=self.sc_mass_arr,
                cr=self.cr_arr,
                srp_area=self.srp_area_arr,
            )
        return self._state_data_cache
    
    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.state_ecef.shape[0]
    
    @property
    def n_unique_ids(self) -> int:
        return len(list(self.unique_ids))

    def __repr__(self) -> str:
        return (
            f"Spacecraft(N={self.N}, n_unique_ids={self.n_unique_ids}, "
            f"start_time={self.stime[0]})"
        )
    
    # Can we add a quick check to the cache to see if things have changed?
    # I also don't have a _commit_state(). Can I instead use a _clear_cache() 
    # and call that at the begining of load_from_file and load_from_array 

