import pyshtools
import numba
import numpy as np


def read_icgem_coeff(file_path: str, encoding: str ="ISO-8859-1"):
    """Read the ICGEM gravitational field model coefficients.

    Parameters
    ----------
    file_path : str or path like object
        Path to ICGEM global earth data model coeffecients 
    encoding : str, optional
        Encoding for reading the files, by default "ISO-8859-1".
        Some files use special characters so a more inclusive encoding is needed.

    Returns
    -------
    clm : 2-D np.array of shape [l,m] holding the c coeffecients for the spherical 
        harmonics
    slm : 2-D np.array of shape [l,m] holding the c coeffecients for the spherical 
        harmonics     
    dictionary containg the meta data from the file
        
    Reference
    ---------
    https://icgem.gfz.de/
    """    
    with open(file_path, "r", encoding=encoding) as potfile:
        #reader = csv.reader(potfile, delimiter=" ", skipinitialspace=True)
        for line in potfile:
            row = line.strip()
            if len(row) < 1:
                continue
            else:
                row = row.split()

            if row[0] == 'gfc':
                l = int(row[1])
                m = int(row[2])
                clm[l,m] = row[3]
                slm[l,m] = row[4]
            elif row[0] == 'max_degree':
                lmax = int(row[1])
                type(lmax)
                clm = np.zeros((lmax+1,lmax+1))
                slm = np.zeros((lmax+1,lmax+1))
            elif row[0] == 'product_type':
                product = row[1]
            elif row[0] == 'radius':
                r0 = float(row[1])
            elif row[0] == 'earth_gravity_constant':
                gm = float(row[1])


    return clm, slm, {'prodcut':product,'r0':r0, 'GM':gm, 'lmax':lmax}





def get_potential(r, lat, lon, clm, slm, gm, r0, lmax=50):
    """
    Calculates gravitational potential using the clm, slm coeffecients passed.
    r: Earth Centered Earth Fixed (meters)
    lat: Earth Centered Earth Fixed latitude (radians)
    lon: Earth Centered Earth Fixed longitude (radians)
    """

    # Calculate r (meters), lat, and lon (radians)

    # Get normalized Legendre functions at the target latitude
    # Note: 'geodesy' normalization is required for EGM96
    theta = np.pi/2 - lat  # Convert latitude (radians) to colatitude (radians)
    # Ensure the Legendre array is explicitly a C-contiguous 2D numpy array for Numba
    p_normalized = pyshtools.legendre.PlmBar(lmax, np.cos(theta))

    # Prepare inputs for numba-jitted function
    rad_ratio = r0 / r

    m_values = np.arange(lmax + 1, dtype=np.float64)
    cos_m_lon = np.cos(m_values * lon)
    sin_m_lon = np.sin(m_values * lon)

    # Call the numba-jitted core function, passing the 2D Legendre array
    potential_sum_core = _get_potential_numba_core(
        lmax, rad_ratio, p_normalized, clm, slm, cos_m_lon, sin_m_lon)

    # Final Potential V = (GM/r) * (1 + sum)
    v = (gm / r) * (1 + potential_sum_core)
    return v

@numba.jit(nopython=True, fastmath=True)
def _get_potential_numba_core(lmax, rad_ratio, p_normalized, 
                              clm_arr, slm_arr, cos_m_lon, sin_m_lon):
    
    potential_sum = 0.0

    for l in range(2, lmax + 1):
        inner_sum = 0.0
        r_l = rad_ratio**l
        for m in range(l + 1):
            c_lm = float(clm_arr[l, m])
            s_lm = float(slm_arr[l, m])

            current_cos_mlon = float(cos_m_lon[m])
            current_sin_mlon = float(sin_m_lon[m])

            p_ind = int(l * (l + 1) / 2 + m)

            # Access the Legendre polynomial from the 1D array
            p_lm = float(p_normalized[p_ind])

            inner_sum += p_lm * (c_lm * current_cos_mlon + s_lm * current_sin_mlon)

        potential_sum += r_l * inner_sum
    return potential_sum

    