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

    Raw inputs may be provided directly (position, time) OR loaded
    from one or more files on disk. Internal state is always strict
    and guaranteed after loading.
    """

    # ---- Raw user inputs (flexible) ----
    position: npt.ArrayLike | None = None
    time: npt.ArrayLike | None = None
    state_file: str | Path | Iterable[str | Path] | None = None

    # ---- Loader configuration ----
    loader: str | None = None  # csv, text, hdf, spk
    time_col: str = "time"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"

    # ---- Internal normalized state (strict) ----
    r_ecef: npt.NDArray[np.float64] = field(init=False)
    stime: pd.Series = field(init=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __post_init__(self):
        """
        Load spacecraft state from either direct inputs or state files.
        Direct in-memory inputs take precedence over file-based loading.
        """
        if self.position is not None or self.time is not None:
            if self.position is None or self.time is None:
                raise ValueError("Both position and time must be provided together")
            self.load_from_arrays(self.position, self.time)
        else:
            if self.state_file is None:
                raise ValueError(
                    "Either (position, time) or state_file must be provided"
                )
            self.load_from_file(self.state_file)

    # ------------------------------------------------------------------
    # Loader / normalizer (array-based)
    # ------------------------------------------------------------------
    def load_from_arrays(
        self,
        position: npt.ArrayLike,
        time: npt.ArrayLike,
    ) -> None:
        """
        Load spacecraft state from in-memory arrays and normalize them
        into internal state variables.
        """
        r = np.asarray(position, dtype=float)
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("position must have shape (N,3)")

        t = pd.to_datetime(time, utc=True)
        if len(t) != r.shape[0]:
            raise ValueError("position and time must have the same length")

        self.r_ecef = r
        self.stime = t

    # ------------------------------------------------------------------
    # Loader / normalizer (file-based)
    # ------------------------------------------------------------------
    def load_from_file(
        self,
        state_file: str | Path | Iterable[str | Path],
    ) -> None:
        """
        Load spacecraft position and time from one or more files and
        normalize them into internal state variables.

        Parameters
        ----------
        state_file : str, Path, iterable, or wildcard
            File(s) containing spacecraft state data
        """

        files = self._expand_files(state_file)

        frames: list[pd.DataFrame] = []
        for file in files:
            df = self._load_table(file)
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)

        for col in (self.time_col, self.x_col, self.y_col, self.z_col):
            if col not in df_all.columns:
                raise ValueError(f"Required column '{col}' not found")

        t = pd.to_datetime(df_all[self.time_col], utc=True)
        r = df_all[[self.x_col, self.y_col, self.z_col]].to_numpy(dtype=float)

        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("Position data must have shape (N,3)")
        if len(t) != r.shape[0]:
            raise ValueError("Time and position columns must have equal length")

        self.r_ecef = r
        self.stime = t

    # ------------------------------------------------------------------
    # Helpers
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

    def _load_table(self, file: Path) -> pd.DataFrame:
        """Dispatch to the appropriate reader based on loader or file extension."""
        suffix = file.suffix.lower()
        loader = self.loader or suffix.lstrip(".")

        if loader in {"csv", "txt", "text"}:
            return pd.read_csv(file)
        elif loader in {"hdf", "h5"}:
            return pd.read_hdf(file)
        elif loader == "spk":
            raise NotImplementedError("SPK loader not yet implemented")
        else:
            raise ValueError(f"Unsupported loader type: {loader}")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        return self.r_ecef.shape[0]

    def __repr__(self) -> str:
        return f"Spacecraft(N={self.N}, start_time={self.stime.iloc[0]})"



