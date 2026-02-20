"""Spacecraft class for deriving energy dissipation and effective density.

added: 19/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import numpy as np
import numpy.typing as npt
import pandas as pd
import glob


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
        if self.state is not None or self.time is not None:
            if self.state is None or self.time is None:
                raise ValueError("Both state and time must be provided together")
            self.load_from_arrays(self.state, self.time, self.sc_id_input)
        else:
            if self.state_file is None:
                raise ValueError("Either (state, time) or state_file must be provided")
            self.load_from_file(self.state_file)

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

        if loader in {"csv", "txt", "text"}:
            return pd.read_csv(file, **kwargs)
        elif loader in {"hdf", "h5"}:
            return pd.read_hdf(file, **kwargs)
        elif loader == "spk":
            raise NotImplementedError("SPK loader not yet implemented")
        else:
            raise ValueError(f"Unsupported loader type: {loader}")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.state_ecef.shape[0]

    def __repr__(self) -> str:
        return (
            f"Spacecraft(N={self.N}, unique_ids={list(self.unique_ids)}, "
            f"start_time={self.stime[0]})"
        )
