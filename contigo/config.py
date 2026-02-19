"""Config and State values for CONTIGO

added: 19/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""

from os import makedirs
from pathlib import Path


def data_path():
    # setup data dir 
    # find directory of module
    # module directory/swdata/ is where the data is stored
    f_path = Path(__file__).resolve()
    d_dir = f_path / '..' / 'data'
    d_dir = d_dir.resolve()

    # create it if it doesn't exist
    if not d_dir.exists():
        makedirs(d_dir)

    return d_dir._str

# Static configuration values (constants)
DATA_DIR = data_path()

# Mutable State Values
state = {'kernel_downloaded':False,
         'pot_coef_loaded':False,
         'pot_file': None, 
         'pot_clm': None,
         'pot_slm': None,
         'pot_r0': None,
         'pot_GM': None,}
