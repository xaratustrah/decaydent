#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
reader of HDF5 data

2022 Xaratustrah
"""

import h5py
import matplotlib.cm as cm
import argparse, logging
import numpy as np
import matplotlib.pyplot as plt

def get_item(filename, item_no):
    with h5py.File(filename, 'r') as hdf:
        
        image = hdf['images'][:]
        bboxes_as_image = hdf['bboxes_as_image'][:]
        instances = hdf['instances'][:]
        
        n_items_total = np.shape(image)[2]
        if item_no >= n_items_total:            
            raise ValueError(f'File has only {n_items_total} items in it, possible indexes are between 0 and {n_items_total-1}.')
        
        bboxes_as_ndarray = np.array(hdf[f'bboxes_as_list/{item_no}'])
        decay_coords_as_array = np.array(hdf[f'decay_coords_as_list/{item_no}'])
    return image[:,:,item_no], bboxes_as_image[:,:,item_no], instances[:,:,item_no], bboxes_as_ndarray, decay_coords_as_array


def plot_pattern(filename, image, bboxes, instances):
    fig, axs = plt.subplots(1,3, figsize=(12, 6), dpi=80)
    axs[0].pcolormesh(image, cmap=cm.jet)
    axs[0].set_title('Pattern')
    axs[1].pcolormesh(image + bboxes, cmap=cm.jet)
    axs[1].set_title('Pattern with bounding boxes')
    axs[2].pcolormesh(instances, cmap=cm.jet)
    axs[2].set_title('Instances')
    fig.suptitle('Spill pattern with decays')
    plt.savefig(filename+'.png', dpi=300)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('infilename', nargs=1, type=str,
                    help='Name of the input file.')
    parser.add_argument('-n', '--number', type=int, default=0,
                        help='Which dataset to extract')

    args = parser.parse_args()
    infilename = args.infilename[0]
    
    logger = logging.getLogger(__name__)
    
    try:
        image, bboxes, instances, bboxes_as_ndarray, decay_coords_as_array = get_item(infilename, args.number)
    except Exception as err:
        logger.error(err)
        #raise
        return
    
    plot_pattern(f'{args.infilename[0]}_{args.number:03d}', image, bboxes, instances)
    print(bboxes_as_ndarray)
    print('')
    print(decay_coords_as_array)
    
if __name__ == '__main__':
    main()
