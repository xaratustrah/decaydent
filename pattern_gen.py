#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes for creating simulated decay pattern_images

2022 Xaratustrah
"""

import numpy as np
import sys
import os
import json
import h5py
import argparse
import nibabel as nib
from pathlib import Path
from loguru import logger
from PIL import Image, ImageDraw

XMAX = 640
YMAX = 480


class Decay():
    def __init__(self, decay_seed=200, line_thickness=5, q_value=150, mother_freq=200):
        self.decay_seed = decay_seed
        self.line_thickness = line_thickness
        self.q_value = q_value
        self.mother_freq = mother_freq

    def make_decay_pattern(self):

        child_freq_final = self.mother_freq + self.q_value
        child_freq_init = np.clip(np.random.randint(XMAX), 2, XMAX-2)
        decay_time = int(
            np.clip(np.random.exponential(self.decay_seed), 2, YMAX-2))
        cooling_time = np.clip(decay_time + 30, 2, YMAX-2)

        b = np.zeros(XMAX)
        instance_only = np.zeros(XMAX)

        for i in range(decay_time):
            slice = np.random.normal(
                self.mother_freq, scale=self.line_thickness, size=XMAX)
            slice_histo = np.histogram(slice, bins=XMAX, range=(0, XMAX))[0]
            b = np.vstack((b, slice_histo))
            instance_only = np.vstack((instance_only, np.zeros(XMAX)))

        for j in range(decay_time, cooling_time - 1):
            child_current_pos = Decay.get_cooling_path_linear(
                j, child_freq_init, decay_time, child_freq_final, cooling_time)

            slice = np.random.normal(
                child_current_pos, scale=self.line_thickness, size=XMAX)
            slice_histo = np.histogram(slice, bins=XMAX, range=(0, XMAX))[0]
            b = np.vstack((b, slice_histo))

            slice_for_instance = np.random.normal(
                child_current_pos, scale=self.line_thickness / 2, size=XMAX)
            slice_for_instance_histo = np.histogram(
                slice_for_instance, bins=XMAX, range=(0, XMAX))[0]
            instance_only = np.vstack(
                (instance_only, slice_for_instance_histo))

        for k in range(cooling_time, YMAX):
            slice = np.random.normal(
                child_freq_final, scale=self.line_thickness, size=XMAX)
            slice_histo = np.histogram(slice, bins=XMAX, range=(0, XMAX))[0]
            b = np.vstack((b, slice_histo))
            instance_only = np.vstack((instance_only, np.zeros(XMAX)))

        # x0, y0, x1, y1
        bbox = [child_freq_init, decay_time, child_freq_final, cooling_time]

        # x, y --> decay time
        decay_coord = [decay_time, child_freq_init]

        # set all non zero values of the instance to maximum color
        instance_only[np.nonzero(instance_only)] = 255

        # make sure to clip everything
        return b[:YMAX, :XMAX], instance_only[:YMAX, :XMAX], decay_coord, bbox

    def get_cooling_path_linear(y, x1, y1, x2, y2):
        # linear
        m = (y2-y1)/(x2-x1)
        return int((y-y1+m*x1)/m)


class Spill():
    def __init__(self, max_decays=5):
        self.max_decays = max_decays

    def create_pattern(self, bkgnd=False):
        self.pattern_image = np.zeros((YMAX, XMAX))
        self.pattern_instances = np.zeros((YMAX, XMAX))
        self.pattern_bboxes_as_image = np.zeros((YMAX, XMAX))
        self.pattern_bboxes_as_list = []
        self.pattern_decay_coords_as_list = []

        for i in range(np.random.randint(self.max_decays)):
            decay = Decay()
            try:
                b, instance_only, decay_coords, bbox = decay.make_decay_pattern()
                self.pattern_image += b
                self.pattern_instances += instance_only
                self.pattern_bboxes_as_image += self.draw_bbox(bbox)
                self.pattern_bboxes_as_list.append(bbox)
                self.pattern_decay_coords_as_list.append(decay_coords)
            except:
                logger.info('One miss...')

        if bkgnd:
            self.pattern_image += self.make_noise()

    def make_noise(self):
        # make some noisy background
        a = np.random.randint(50, size=YMAX * XMAX * 3)
        noise = np.reshape(a, (YMAX, XMAX, 3))
        return noise[:, :, 0]

    def draw_bbox(self, bbox):
        out = Image.new("RGB", (XMAX, YMAX), (0, 0, 0))
        # get a drawing context
        d = ImageDraw.Draw(out)
        d.rectangle(bbox, fill=None, outline='#ff0000', width=1)
        # out.show()
        # casting back and forth, other direction Image.fromarray
        # it is possibe to do casting directly
        drawstuff = np.array(out)[:, :, 0]
        return drawstuff


# ------------------------

def print_shape(name, var):
    logger.info(f'Shape of {name} is {np.shape(var)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('outfilename', nargs=1, type=str,
                        help='Name of the output files.')
    parser.add_argument('-n', '--nsim', type=int, default=4,
                        help='Number of simulations')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='output directory.')
    parser.add_argument('--hdf', action='store_true')
    parser.add_argument('--nnunet', action='store_true')

    args = parser.parse_args()
    outfilename = args.outfilename[0]

    if args.outdir:
        # handle trailing slash properly
        outfilepath = os.path.join(args.outdir, '')

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(outfilepath + 'pettern_gen.log', level='DEBUG')

    if args.hdf:
        pattern_images_all = np.zeros((YMAX, XMAX))
        pattern_bboxes_as_image_all = np.zeros((YMAX, XMAX))
        pattern_instances_all = np.zeros((YMAX, XMAX))
        pattern_bboxes_as_list_all = []
        pattern_decay_coords_as_list_all = []

        for i in range(args.nsim):
            logger.info(f"Preparing case{i:05d}")
            spill = Spill()
            spill.create_pattern(bkgnd=True)

            # dstack is equal to stack -1 or 2 for three dim arrays
            pattern_images_all = np.dstack(
                (pattern_images_all, spill.pattern_image))
            pattern_bboxes_as_image_all = np.dstack(
                (pattern_bboxes_as_image_all, spill.pattern_bboxes_as_image))
            pattern_instances_all = np.dstack(
                (pattern_instances_all, spill.pattern_instances))
            pattern_bboxes_as_list_all.append(spill.pattern_bboxes_as_list)
            pattern_decay_coords_as_list_all.append(
                spill.pattern_decay_coords_as_list)

        with h5py.File(f'{outfilepath}{outfilename}.hdf5', 'w') as hf:

            hf.create_dataset(
                'images', data=pattern_images_all[:, :, 1:], compression="gzip")
            hf.create_dataset(
                'bboxes_as_image', data=pattern_bboxes_as_image_all[:, :, 1:], compression="gzip")
            hf.create_dataset('instances', data=pattern_instances_all.astype(bool)[
                              :, :, 1:], compression="gzip")
            for ii in range(len(pattern_decay_coords_as_list_all)):
                hf.create_dataset(
                    f'bboxes_as_list/{ii}', data=pattern_bboxes_as_list_all[ii], compression="gzip")
                hf.create_dataset(
                    f'decay_coords_as_list/{ii}', data=pattern_decay_coords_as_list_all[ii], compression="gzip")

    elif args.nnunet:
        task_name = f'Task100_{outfilename}'
        base_path = f'{outfilepath}{task_name}'
        images_path = f'{base_path}/raw_splitted/imagesTr'
        labels_path = f'{base_path}/raw_splitted/labelsTr'

        Path(images_path).mkdir(parents=True, exist_ok=True)
        Path(labels_path).mkdir(parents=True, exist_ok=True)

        with open(f'{base_path}/dataset.json', 'w') as f:
            json.dump({
                'task': task_name,
                'name': outfilename,
                'dim': 3,
                'test_labels': True,
                'labels': {'0': 'Decay'},
                'modalities': {'0': 'ESR'},
            }, f, indent=4)

        for i in range(args.nsim):
            logger.info(f"Preparing case{i:05d}")
            spill = Spill()
            spill.create_pattern(bkgnd=True)
            b = np.expand_dims(spill.pattern_image, axis=2)
            # normalize to 1
            b = b/b.max()
            new_image = nib.Nifti1Image(b, affine=np.eye(4))
            nib.save(new_image, f'{images_path}/case{i:05d}_00.nii.gz')

            c = np.expand_dims(spill.pattern_instances, axis=2)
            # normalize to 1
            c[np.nonzero(c)] = 1
            new_instance = nib.Nifti1Image(c, affine=np.eye(4))
            nib.save(new_instance, f'{labels_path}/case{i:05d}.nii.gz')

            with open(f'{labels_path}/case{i:05d}.json', 'w') as f:
                json.dump({'instances': {'1': 0}}, f, indent=4)

    else:
        logger.info('Nothing to do. Exiting.')


if __name__ == '__main__':
    main()
