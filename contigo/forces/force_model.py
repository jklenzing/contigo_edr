@runtime_checkable
class ForceModel(Protocol):
    """
    Protocol that all force models must follow.

    Required:
        acceleration(spacecraft) -> (N,3)

    Optional (for conservative forces):
        potential(spacecraft) -> (N,)

    Attributes:
        name: str
        is_conservative: bool
    """

    name: str
    is_conservative: bool

    def acceleration(self, spacecraft: Spacecraft) -> npt.NDArray[np.float64]:
        ...

    def potential(self, spacecraft: Spacecraft) -> npt.NDArray[np.float64]:
        ...