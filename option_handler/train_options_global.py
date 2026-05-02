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

import argparse
from option_handler.base_train_options import BaseTrainOptions, json_dict


PRE_TRAINING_POINT_MASK_WEIGHTS = {
    'w_point_face': 1.0,
    'w_point_ears': 1.0,
    'w_point_eyeballs': 1.0,
    'w_point_eye_region': 1.0,
    'w_point_lips': 1.0,
    'w_point_neck': 1.0,
    'w_point_nostrils': 1.0,
    'w_point_scalp': 1.0,
    'w_point_boundary': 1.0,
}

EDGE_MASK_WEIGHTS = {
    'w_edge_face': 2.0,
    'w_edge_ears': 3.0,
    'w_edge_eyeballs': 50.0,
    'w_edge_eye_region': 10.0,
    'w_edge_lips': 3.0,
    'w_edge_neck': 1.0,
    'w_edge_nostrils': 10.0,
    'w_edge_scalp': 10.0,
    'w_edge_boundary': 10.0,
}

DENSE_LANDMARKS_WEIGHTS = {
    'face': 0.0,
    'ears': 0.0,
    'eyeballs': 0.0,
    'eye_region': 1.0,
    'lips': 1.0,
    'neck': 0.0,
    'nostrils': 0.0,
    'scalp': 0.0,
    'boundary': 0.0,
    'left_eyeball': 0.0,
    'right_eyeball': 0.0,
    'right_ear': 0.0,
    'right_eye_region': 0.0,
    'left_ear': 0.0,
    'left_eye_region': 0.0,
}


class TrainOptions(BaseTrainOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseTrainOptions.initialize(self)
        self.isTrain = True
        return self.parser

    def initialize_extra(self):
        # base
        self.add_arg(cate='base', abbr='cf',  name='config-filename', type=str, default='')
        self.add_arg(cate='base', abbr='md',  name='model-directory', type=str, default='./runs')
        self.add_arg(cate='base', abbr='eid', name='experiment-id', type=str, default='TEMPEH')

        # data
        self.add_arg(cate='data', abbr='num-spnts', name='number-sample-points', type=list, default=[5023])
        self.add_arg(cate='data', abbr='templ-name', name='template-fname', type=str, default='./assets/template/sampling_template.obj')
        self.add_arg(cate='data', abbr='vmask-name', name='vertex-mask-fname', type=str, default='./assets/template/vertex_masks2.npz')
        self.add_arg(cate='data', abbr='b-sigma', name='brightness-sigma', type=float, default=0.33)
        self.add_arg(cate='data', abbr='v_scan', name='scan-vertex-count', type=int, default=20000)

        self.add_arg(cate='data', abbr='tdl', name='train-data-list-fname', type=str, default='')
        self.add_arg(cate='data', abbr='vdl', name='val-data-list-fname', type=str, default='')
        self.add_arg(cate='data', abbr='data-dir', name='dataset-directory', type=str, default='')
        self.add_arg(cate='data', abbr='scan-dir', name='scan-directory', type=str, default='')
        self.add_arg(cate='data', abbr='reg-dir', name='processed-directory', type=str, default='')
        self.add_arg(cate='data', abbr='image-dir', name='image-directory', type=str, default='')
        self.add_arg(cate='data', abbr='normals-image-dir', name='normals-image-directory', type=str, default='')
        self.add_arg(cate='data', abbr='depths-dir', name='depths-image-directory', type=str, default='')
        self.add_arg(cate='data', abbr='image-ext', name='image-file-ext', type=str, default='png')
        self.add_arg(cate='data', abbr='calib-dir', name='calibration-directory', type=str, default='')
        self.add_arg(cate='data', abbr='dense-dir', name='dense-landmarks-dir', type=str, default='')
        self.add_arg(cate='data', abbr='dense-s-dir', name='dense-semantic-landmarks-dir', type=str, default='')

        # train
        self.add_arg(cate='train', abbr='niter',  name='num-iterations', type=int, default=argparse.SUPPRESS)
        self.add_arg(cate='train', abbr='print-freq',  name='print-frequency', type=int, default=10)
        self.add_arg(cate='train', abbr='vis-freq',  name='visualize-frequency', type=int, default=100)
        self.add_arg(cate='train', abbr='val-freq',  name='validate-frequency', type=int, default=5000)
        self.add_arg(cate='train', abbr='save-freq',  name='save-frequency', type=int, default=1000)

        # model
        self.add_arg(cate='model', abbr='wandb', name='wandb', type=bool, default=False)
        self.add_arg(cate='model', abbr='irf', name='image-resize-factor', type=int, default=1)
        self.add_arg(cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=8)

        self.add_arg(cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')
        self.add_arg(cate='model', abbr='pretr-path-local', name='pretrained-local-path', type=str, default='')
        self.add_arg(cate='model', abbr='pretr-fuse', name='pretrained-fuse', type=str, default='')

        # volumetric sparse point net
        self.add_arg(cate='model', abbr='gvd', name='global-voxel-dim', type=int, default=32)
        self.add_arg(cate='model', abbr='gvi', name='global-voxel-inc', type=float, default=25.0)
        self.add_arg(cate='model', abbr='go', name='global-origin', type=list, default=[0.0, 0.0, 0.0])
        self.add_arg(cate='model', abbr='nm', name='norm', type=str, default="bn")

        # transformer (rigid_transformer hardcoded)
        self.add_arg(cate='model', abbr='gtrafo-dim', name='global-spatial-transformer-dim', type=int, default=32)
        self.add_arg(cate='model', abbr='gtrafo-levels', name='global-transformer-levels', type=int, default=2)
        self.add_arg(cate='model', abbr='gtrafo-sfactor', name='global-transformer-scale-factor', type=float, default=0.4)
        self.add_arg(cate='model', abbr='gtrafo-pooling', name='global-transformer-use-pooling', type=bool, default=True)

        # loss weights
        self.add_arg(cate='model', abbr='wpointr', name='weight-points-recon', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wpointm', name='point-mask-weights', type=json_dict, default=argparse.SUPPRESS)
        self.add_arg(cate='model', abbr='wp2s', name='weight-points2surface', type=float, default=argparse.SUPPRESS)
        self.add_arg(cate='model', abbr='gmo', name='gmo_sigma', type=float, default=10.0)
        self.add_arg(cate='model', abbr='wedge', name='weight-edge-regularizer', type=float, default=argparse.SUPPRESS)
        self.add_arg(cate='model', abbr='wedgem', name='edge-mask-weights', type=dict, default=EDGE_MASK_WEIGHTS)
        self.add_arg(cate='model', abbr='wdensem', name='dense-mask-weights', type=dict, default=DENSE_LANDMARKS_WEIGHTS)
        self.add_arg(cate='model', abbr='wng', name='weight-normals-grad', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wland', name='weight-landmarks', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wlandd', name='weight-dense-landmarks', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wnorm', name='weight-normals-images', type=float, default=0.0)
        self.add_arg(cate='model', abbr='add-norm', name='add-norm-as-input', type=bool, default=False)
        self.add_arg(cate='model', abbr='wpointmaps', name='weight-point-maps', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wdepthmaps', name='weight-depth-maps', type=float, default=0.0)

        # SMIRK loss (optional perceptual loss head)
        self.add_arg(cate='model', abbr='smirk', name='enable-smirk-loss', type=bool, default=False)
        self.add_arg(cate='model', abbr='wsmirk', name='weight-smirk-loss', type=float, default=0.0)

        self.add_arg(cate='model', abbr='eval', name='evaluate', type=bool, default=False)
        self.add_arg(cate='model', abbr='difr', name='enable-diff-rendering', type=bool, default=False)

        # PLIKS-based FLAME refinement
        self.add_arg(cate='model', abbr='pliks', name='use-pliks-refinement', type=bool, default=False)
        self.add_arg(cate='model', abbr='wvertregpliks', name='weight-vertices-regularizer-pliks', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wvertregpliks-edge', name='weight-vertices-regularizer-pliks-edge', type=float, default=0.0)

        # FLAME pose/shape regularizers
        self.add_arg(cate='model', abbr='wexpreg', name='weight-expression-regularization', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wshapereg', name='weight-shape-regularization', type=float, default=0.0)
        self.add_arg(cate='model', abbr='wposereg', name='weight-pose-regularization', type=float, default=0.0)

        self.add_arg(cate='model', abbr='meters', name='to-meters', type=bool, default=False)

        # local model
        self.add_arg(cate='lmodel', abbr='local', name='enable-local', type=bool, default=False)
        self.add_arg(cate='lmodel', abbr='lvd', name='local-voxel-dim', type=int, default=8)
        self.add_arg(cate='lmodel', abbr='lvi', name='local-voxel-inc-list', type=list, default=[2.0])

        self.add_arg(cate='model', abbr='global', name='global-step-cont', type=int, default=0)

        # refinement (TTO) — used by trainer.train_refine
        self.add_arg(cate='model', abbr='refine-steps', name='refinement-steps', type=int, default=5)
        self.add_arg(cate='model', abbr='refine-lr', name='refinement-lr', type=float, default=1e-4)
        self.add_arg(cate='model', abbr='refine-v', name='refine-vis', type=bool, default=False)
        self.add_arg(cate='model', abbr='refine-vis-freq', name='refine-visualization-freq', type=int, default=1)
        self.add_arg(cate='model', abbr='refine-start', name='refine-start-index', type=int, default=0)
        self.add_arg(cate='model', abbr='refine-end', name='refine-end-index', type=int, default=-1)

        self.initialize_default_parameters()

    def initialize_default_parameters(self):
        self.default_parameters = {
            'num-iterations': 300000,
            'weight-points2surface': 0.0,
            'weight-edge-regularizer': 0.0,
            'point-mask-weights': PRE_TRAINING_POINT_MASK_WEIGHTS,
        }
