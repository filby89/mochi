"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import os
from os.path import join
import time
import math
import numpy as np
import argparse
import imageio
from glob import glob
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from utils.mesh_sampling import MeshSampler
from psbody.mesh import Mesh

from utils.utils import get_filename

# -----------------------------------------------------------------------------

class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.global_step = self.args.global_step_cont

    def initialize(self):
        self.control_seeds()
        self.mkdirs()
        self.register_mesh_sampler()
        self.register_model()
        self.register_losses()
        # self.register_optimizer()
        # self.resume_checkpoint()
        self.register_dataset()
        # self.register_logger()

    # ----------------------
    # meta

    def control_seeds(self):
        # torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # torch.set_float32_matmul_precision('high')

    def mkdirs(self):
        if self.args.model_directory == '': raise RuntimeError(f'invalid model_directory = {self.args.model_directory}')
        if self.args.experiment_id == '': raise RuntimeError(f'invalid experiment_id = {self.args.experiment_id}')
        self.directory_output = join(self.args.model_directory, self.args.experiment_id)
        os.makedirs(self.directory_output, exist_ok=True)

        self.model_dir = join(self.directory_output, 'checkpoints')
        os.makedirs(self.model_dir, exist_ok=True)

    # ----------------------
    # model

    def register_mesh_sampler(self):
        print('Registering mesh sampler...')
        mesh_sampler_fname = join(self.directory_output, 'mesh_sampler.npz')
        if os.path.exists(mesh_sampler_fname):
            self.mesh_sampler = MeshSampler()
            self.mesh_sampler.load(mesh_sampler_fname)
        else:
            template_fname = self.args.template_fname
            if not os.path.exists(template_fname):
                raise RuntimeError('Template not found %s' % template_fname)
            template_mesh = Mesh(filename=template_fname)
            template_mesh.v[:] *= 1000
            mesh_dimension_list = self.args.number_sample_points
            self.mesh_sampler = MeshSampler(template_mesh, mesh_dimension_list, keep_boundary_adjacent=True)
            self.mesh_sampler.save(mesh_sampler_fname)


    def save_checkpoint(self):
        
        if not getattr(self.args, 'flame_only', False):
            model_path = os.path.join(self.model_dir, 'model_%08d.pth' % (self.global_step))
            torch.save({
                'model': self.model.module.state_dict(),
                'optimizer_model': self.optimizer_model.state_dict()
                }, model_path)

        if self.args.enable_smirk_loss:
            fuse_generator_path = os.path.join(self.model_dir, 'fuse_generator_%08d.pth' % (self.global_step))
            torch.save({
                'model': self.fuse_generator.state_dict()
            }, fuse_generator_path)

        if self.args.enable_discriminator:
            discriminator_path = os.path.join(self.model_dir, 'discriminator_%08d.pth' % (self.global_step))
            torch.save({
                'model': self.discriminator.state_dict()
            }, discriminator_path)

        if self.args.enable_flame_branch:
            flame_path = os.path.join(self.model_dir, 'flame_%08d.pth' % (self.global_step))
            torch.save({
                'model': self.flame_model.state_dict()
            }, flame_path)
        
        if self.args.enable_local:
            flame_path = os.path.join(self.model_dir, 'local_%08d.pth' % (self.global_step))
            torch.save({
                'model': self.local_model.state_dict()
            }, flame_path)

    def cleanup_memory(self):
        """Comprehensive memory cleanup"""
        # Clear per-pixel loss tensors
        attrs_to_clear = [attr for attr in dir(self) if any(x in attr for x in [
            'loss_per_pixel', 'maps_pred', 'fused_images', 'masked_images'
        ])]
        
        for attr in attrs_to_clear:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Clear data references
        self.data = None
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def worker_init_fn(self, worker_id):
        # to properly randomize:
        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def make_data_loader(self, dataset, cuda=True, shuffle=True):
        from torch.utils.data._utils.collate import default_collate

        def custom_collate_fn(batch):
            collated = {}
            keys = batch[0].keys()
            
            for key in keys:
                if key in ("v_scan", "f_scan"):
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = default_collate([item[key] for item in batch])

            return collated

        kwargs = {'num_workers': self.args.thread_num, 'pin_memory': True} if cuda else {}
        return torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle, drop_last=True, prefetch_factor=4,
                                            worker_init_fn=self.worker_init_fn, collate_fn=custom_collate_fn, **kwargs)



    def load_flame_masks(self):
        self.flame_masks_triangles = np.load('assets/FLAME_masks/FLAME_masks_triangles.npy', allow_pickle=True).item()
        # dict_keys(['eye_region', 'neck', 'left_eyeball', 'right_eyeball', 'right_ear', 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region', 'face_clean', 'cleaner_lips'])
        self.flame_masks_triangles.pop('face_clean')  # remove 'face_clean' as it is not used
        self.flame_masks_triangles.pop('cleaner_lips')  # remove 'face_clean' as it is not used
        self.flame_masks_triangles.pop('eye_region')  # remove 'face_clean' as it is not used
        # merge ears
        self.flame_masks_triangles['ears'] = np.concatenate([
            self.flame_masks_triangles['left_ear'],
            self.flame_masks_triangles['right_ear']
        ])
        # merge eyes
        self.flame_masks_triangles['eyes'] = np.concatenate([
            self.flame_masks_triangles['left_eye_region'],
            self.flame_masks_triangles['right_eye_region']
        ])
        self.flame_masks_triangles['eyeballs'] = np.concatenate([
            self.flame_masks_triangles['left_eyeball'],
            self.flame_masks_triangles['right_eyeball']
        ])
        self.flame_masks_triangles.pop('left_ear')
        self.flame_masks_triangles.pop('right_ear')
        self.flame_masks_triangles.pop('left_eye_region')
        self.flame_masks_triangles.pop('right_eye_region')
        self.flame_masks_triangles.pop('left_eyeball')
        self.flame_masks_triangles.pop('right_eyeball')
        print('Regions in FLAME masks:', self.flame_masks_triangles.keys())

        import pickle
        flame_masks = pickle.load(
                    open('assets/FLAME_masks/FLAME_masks.pkl', 'rb'),
                    encoding='latin1')
        self.flame_masks = flame_masks


        self.vertex_masks_tempeh = np.load("assets/template/vertex_masks2.npz")






    # ----------------------
    # main processes

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def run(self):
        pass