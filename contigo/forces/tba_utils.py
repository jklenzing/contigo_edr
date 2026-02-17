from math import sqrt
import numba


@numba.jit(nopython=True,parallel=True, fastmath=True)
def tba_pairwise_numba(r_sat, r_bodies, mu_bodies):
    """
    Per-body third-body acceleration.

    Parameters
    ----------
    r_sat : (N,3) array
        Satellite position vectors
    r_bodies : (B,N,3) array
        Third-body position vectors
    mu_bodies : (B,) array
        Gravitational parameters

    Returns
    -------
    a : (B,N,3) array
        Acceleration due to each body
    """

    B, N, _ = r_bodies.shape
    a = np.zeros_like(r_bodies)

    for i in numba.prange(N):
        sx = r_sat[i, 0]
        sy = r_sat[i, 1]
        sz = r_sat[i, 2]

        for b in range(B):
            bx = r_bodies[b, i, 0]
            by = r_bodies[b, i, 1]
            bz = r_bodies[b, i, 2]

            dx = bx - sx
            dy = by - sy
            dz = bz - sz

            rho2 = dx*dx + dy*dy + dz*dz
            rb2  = bx*bx + by*by + bz*bz

            inv_rho3 = 1.0 / (sqrt(rho2) * rho2)
            inv_rb3  = 1.0 / (sqrt(rb2)  * rb2)

            mu = mu_bodies[b]

            a[b, i, 0] = mu * (dx * inv_rho3 - bx * inv_rb3)
            a[b, i, 1] = mu * (dy * inv_rho3 - by * inv_rb3)
            a[b, i, 2] = mu * (dz * inv_rho3 - bz * inv_rb3)

    return a