"""Utilites for the contigo module

added: 17/02/2026 Kyle Murphy <kylemurphy.spacephys@gmail.com>
"""
import pathlib
import functools
import shutil
import posixpath
import urllib.parse
import logging
import requests

import numpy as np
import numpy.typing as npt

from os import path
from datetime import datetime
from dateutil import tz
from tqdm import tqdm

import spiceypy as spice

import contigo.config as config
import pandas as pd

logger = logging.getLogger(__name__)


def dl_file(url, filename):
    """
    Download file from url to filename

    Parameters
    ----------
    url : url to file.
    filename : filename to save as.

    Raises
    ------
    for
        DESCRIPTION.
    RuntimeError
        DESCRIPTION.

    """
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    
    # if requests couldn't get file size drop encoding to get it
    if file_size == 0:
        rs = requests.head(url, headers={'Accept-Encoding': None})
        file_size = int(rs.headers.get('Content-Length', 0))
        
    
    fpath = pathlib.Path(filename).expanduser().resolve()
    fpath.parent.mkdir(parents=True, exist_ok=True)
    
    desc = "(Unknown total file size)" if file_size == 0 else ""
    post = f'Downloading: {url} to {filename}'
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc, postfix=post, position=0, leave=True) as r_raw:
        with fpath.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
            
def wf_mtime(url):
    """
    Retrieves the last modified date of a file on the web.

    Args:
        url (str): The URL of the file.

    Returns:
        datetime or None: The last modified date as a datetime object,
                          or None if the header is not found or an error occurs.
    """
    try:
        response = requests.head(url)  # Use HEAD request for efficiency
        response.raise_for_status()  # Raise an exception for bad status codes

        last_modified_header = response.headers.get('Last-Modified')

        if last_modified_header:
            # Parse the date string from the header
            # Example format: 'Wed, 21 Oct 2015 07:28:00 GMT'
            return datetime.strptime(
                last_modified_header, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=tz.gettz('GMT'))
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    
def df_sp3(fn):
    """Read in SP3c ASCII text file.

    Parameters
    ----------
    fn : function
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    
    dt = []
    dat = []
    vel = []

    with open(fn, 'r') as f:
        for line in f:
            # Implement parsing logic based on SP3 format
            # Example: Check for specific record types like header records,
            # position records, or clock records and extract data accordingly.
            if line.startswith('*'): # Example for epoch records
                # Parse epoch information
                dt.append(line[1:-1].strip())
                
            elif line.startswith('P'): # Example for position records
                # Parse satellite position and clock information
                dat.append(line.split()[0:4])
            elif line.startswith('V'):
                vel.append([line[4:18],line[18:32],line[32:46]])


    dt = pd.to_datetime(dt,format='%Y %m %d %H %M %S.%f')

    pdf = pd.DataFrame(data=dat,columns=['sat','x','y','z'])
    pdf[['x','y','z']] = pdf[['x','y','z']].astype(float)
    pdf['time'] = dt

    vdf = pd.DataFrame(data=vel,columns=['vx','vy','vz'])
    vdf[['vx','vy','vz']] = vdf[['vx','vy','vz']].astype(float)
    vdf[['vx','vy','vz']] = vdf[['vx','vy','vz']]/10000.

    sdf = pd.merge(pdf, vdf, left_index=True, right_index=True)

    return sdf

def spice_et(stime: npt.ArrayLike, tscale: str):

    allowed = {'GPS', 'TAI', 'UTC', 'ET', 'TDB'}
    tscale = tscale.upper()
    if tscale not in allowed:
        raise ValueError(f"tscale must be one of {allowed}")

    # get the leapsecond kernel and make sure it's
    # loaded, download it if we don't have it
    leaps_f = config.LEAP_FILE
    lp_kernel = path.join(config.DATA_DIR,leaps_f)

    sp_kcnt = spice.ktotal('ALL')
    sp_loaded = [spice.kdata(i,'ALL')[0] for i in range(sp_kcnt)]
    if path.exists(lp_kernel) and [lp_kernel] not in sp_loaded:
        spice.furnsh(lp_kernel) # need to check if kernels are loaded
    else:
        base_url = 'https://naif.jpl.nasa.gov'
        base_pth = '/pub/naif/generic_kernels'
        leaps_d = 'lsk'

        leaps_url = urllib.parse.urljoin(base_url,
                                      posixpath.join(base_pth,leaps_d,leaps_f))

        logger.info('Downloading kernel - %s', lp_kernel)
        dl_file(leaps_url, lp_kernel)
        spice.furnsh(lp_kernel)

    if tscale == 'UTC':
        t_str = pd.to_datetime(np.array(stime)).strftime('%d %b %Y %H:%M:%S.%f')
        et = np.array([spice.utc2et(sp_in) for sp_in in t_str]) 
    else:
        j2000 = pd.Timestamp('2000-01-01 12:00:00')
        spj2000 = ((stime - j2000).dt.total_seconds()).to_list()
        et = np.array([spice.unitim(sp_in,tscale,'ET') for sp_in in spj2000])

    return et