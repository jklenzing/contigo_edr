import os

from pathlib import Path
from typing import List, Union, Optional


import jpype
import orekit_jpype as orekit

import contigo.config as config

def start_orekit(vmargs: Union[str, None] = None,
           additional_classpaths: Union[List, None] = None,
           jvmpath: Optional[Union[str, os.PathLike]] = None):


    if jpype.isJVMStarted() is False:
        print('Starting Orekit JVM')
        # get the base directory so we can find files
        f_path = Path(__file__).resolve()
        base_dir = f_path / '..' / '..' / '..'
        if additional_classpaths is None:

            d_dir = base_dir / 'java_src' / 'target' / 'orekit_utils-1.0.0.jar'
            d_dir = d_dir.resolve()

            additional_classpaths = [d_dir]

        #start the orekit JVM with the required
        #contigo class path
        orekit.initVM(jvmpath=jvmpath,
              additional_classpaths=additional_classpaths)

        # finish seeting up the orekit data
        from orekit_jpype.pyhelpers import setup_orekit_data
        from orekit_jpype.pyhelpers import download_orekit_data_curdir

        # check for the orekit data file
        orekit_data = Path(config.DATA_DIR).resolve() / 'orekit_data.zip'
        orekit_data = orekit_data.resolve()
        if not orekit_data.exists():
            print(f'Downloading Orekit data to {orekit_data}')
            download_orekit_data_curdir(orekit_data._str)

        # setup the orekit data
        print(f'Loading Orekit data to {orekit_data}')
        setup_orekit_data(filenames=orekit_data._str, from_pip_library=False)

        # set the state variable to true so we know orekit has been loaded
        config.state['orekit_loaded'] = True
