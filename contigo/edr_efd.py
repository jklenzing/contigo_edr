# ==================================================================
# Energy Dissipation Rate Density (EDRDensity)
# ==================================================================
class EDRDensity:
    """
    Energy Dissipation Rate Density calculator.

    Accepts:
        - Spacecraft OR Constellation
        - list of ForceModel implementations

    Computes:
        - Total acceleration
        - Conservative potential (if available)
        - Energy dissipation rate (v · a_nonconservative)
    """

    def __init__(
        self,
        system: Spacecraft | Constellation,
        force_models: list[ForceModel],
    ):
        self.system = system
        self.force_models = force_models

    # --------------------------------------------------------------
    def compute(self) -> dict:
        """
        Returns dictionary keyed by spacecraft ID.
        """

        if isinstance(self.system, Spacecraft):
            spacecraft_dict = {self.system.unique_ids[0]: self.system}
        else:
            spacecraft_dict = self.system.spacecraft

        results = {}

        for sc_id, sc in spacecraft_dict.items():
            N = sc.N
            v = sc.state_ecef[:, 3:6]

            total_acc = np.zeros((N, 3))
            noncon_acc = np.zeros((N, 3))
            total_potential = np.zeros(N)

            for model in self.force_models:
                acc = model.acceleration(sc)
                total_acc += acc

                if getattr(model, "is_conservative", False):
                    if hasattr(model, "potential"):
                        total_potential += model.potential(sc)
                else:
                    noncon_acc += acc

            # Energy dissipation rate density: v · a_nonconservative
            edr = np.einsum("ij,ij->i", v, noncon_acc)

            results[sc_id] = {
                "total_acceleration": total_acc,
                "nonconservative_acceleration": noncon_acc,
                "potential": total_potential,
                "edr": edr,
            }

        return results
