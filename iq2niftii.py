#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert IQ data to NIFTII format

2022 Xaratustrah
"""

from iqtools import *
import matplotlib.pyplot as plt
import nibabel as nib
from loguru import logger
import os
from pathlib import Path 

import sys

START=250
STOP = 900
SFRAMES = 200
LFRAMES = 1024

def main():
    for filename in sys.argv[1:]:
        logger.info(f"Preparing {Path(filename).name}")
        iq = get_iq_object(filename)
        iq.read(nframes=1, lframes=LFRAMES)
        nframes = int(iq.nsamples_total / LFRAMES - SFRAMES)
        iq.read(nframes=nframes, lframes=LFRAMES, sframes=SFRAMES)
        iq.method='mtm'
        xx, yy, zz = iq.get_power_spectrogram(nframes=nframes, lframes=LFRAMES)        
        zzary = np.array_split(zz, 4)
        for i in range(len(zzary)):
            c = zzary[i][:,START:STOP]
            write_spectrogram_to_nifti(c, f'{Path(filename).stem}_{i}_0000.nii.gz')

# ---------

if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(f'logger.log', level='DEBUG')

    main()
