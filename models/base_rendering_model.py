import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import format as fmt
import os
from .base_model import BaseModel

from .rendering.diff_render_func import find_render_function, find_blend_function, find_tone_map, alpha_blend
from .rendering.diff_ray_marching import find_ray_generation_method, find_refined_ray_generation_method, \
    ray_march, alpha_ray_march # brdf_render
from utils import format as fmt
from utils.spherical import SphericalHarm, SphericalHarm_table
from utils.util import add_property2dict
from torch.autograd import Variable

from PIL import Image
def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

class BaseRenderingModel(BaseModel):
    ''' A base rendering model that provides the basic loss functions, 
        selctions of different rendering functions, ray generation functions, 
        blending functions (for collocated and non-collocated ray marching), 
        and functions to setup encoder and decoders. 
        A sub model needs to at least re-implement create_network_models() and run_network_models() for actual rendering.
        Examples are: hirarchical_volumetric_model etc.

        The model collects 
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # loss parameters
        parser.add_argument(
            "--sparse_loss_weight",
            type=float,
            default=0,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--color_loss_items",
            type=str,
            nargs='+',
            default=None,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--test_color_loss_items",
            type=str,
            nargs='+',
            default=None,
            help="The (multiple) output items to supervise with gt color.")
        parser.add_argument(
            "--color_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each color supervision item. The number of this args should be 1 or match the number in --color_loss_items"
        )
        parser.add_argument("--color_loss", type=str, default="l2")
        parser.add_argument(
            "--bg_loss_items",
            type=str,
            nargs='+',
            default=[],
            help="The (multiple) output items to supervise with gt masks.")
        parser.add_argument(
            "--bg_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each mask supervision item. The number of this args should be 1 or match the number in --bg_loss_items"
        )
        parser.add_argument(
            "--depth_loss_items",
            type=str,
            nargs='+',
            default=[],
            help="The (multiple) output items to supervise with gt depth.")
        parser.add_argument(
            "--depth_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each depth supervision item. The number of this args should be 1 or match the number in --depth_loss_items"
        )
        parser.add_argument(
            "--zero_one_loss_items",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to regularize to be close to either 0 or 1 ."
        )
        parser.add_argument(
            "--zero_one_loss_weights",
            type=float,
            nargs='+',
            default=[1.0],
            help=
            "The weights for each zero_one regularization item. The number of this args should be 1 or match the number in --zero_one_loss_items"
        )

        parser.add_argument(
            "--l2_size_loss_items",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to regularize to be close to either 0 or 1 ."
        )
        parser.add_argument(
            "--l2_size_loss_weights",
            type=float,
            nargs='+',
            default=[0.0],
            help=
            "The weights for each zero_one regularization item. The number of this args should be 1 or match the number in --zero_one_loss_items"
        )
        parser.add_argument(
            "--zero_epsilon",
            type=float,
            default=1e-3,
            help="epsilon in logarithmic regularization terms when needed.",
        )
        parser.add_argument(
            "--no_loss",
            type=int,
            default=False,
            help="do not compute loss.",
        )
        parser.add_argument("--eik_loss_weight", type=float, default=0)
        parser.add_argument("--point_normal_weight", type=float, default=0)
        parser.add_argument("--orientation_weight", type=float, default=0)
        parser.add_argument("--relsdf_loss_weight", type=float, default=0)
        parser.add_argument("--relsdf_loss_items", type=str, nargs='+',
            choices=['back_sdf', 'est_delta', 'dist_bound'], default=['back_sdf'])
        parser.add_argument("--relsdf_delta_thres", type=float, default=0.01)
        parser.add_argument("--relsdf_w_bias", type=float, default=0.1)
        parser.add_argument("--relsdf_back_sdf_weight", type=float, default=1.0)
        parser.add_argument("--relsdf_single_dir", action='store_true')
        parser.add_argument("--relsdf_uni_weight", action='store_true')
        parser.add_argument("--relsdf_normal_type", type=str, choices=['agg', 'pred'], default='agg')
        # parser.add_argument("--ref_normal_reg", action='store_true')
        parser.add_argument("--ref_weight_detach", action='store_true')
        
        parser.add_argument("--light_tv_weight", type=float, default=0)
        parser.add_argument("--light_achro_weight", type=float, default=0)
        parser.add_argument("--diffuse_tv_weight", type=float, default=0)
        parser.add_argument("--normal_tv_weight", type=float, default=0)
        parser.add_argument("--roughness_tv_weight", type=float, default=0)
        parser.add_argument("--roughness_reg_weight", type=float, default=0)
        parser.add_argument("--feat_tv_weight", type=float, default=0)
        parser.add_argument("--sdf_tv_weight", type=float, default=0)
        parser.add_argument("--density_distill_weight", type=float, default=0)
        parser.add_argument("--light_reg_weight", type=float, default=0)
        parser.add_argument("--sdf_center_weight", type=float, default=0)
        parser.add_argument("--vis_map_bce_weight", type=float, default=0)
        parser.add_argument("--pnt_diffuse_weight", type=float, default=0)
        parser.add_argument("--mlp_diffuse_cor_weight", type=float, default=0)
        parser.add_argument("--sparse_loss_mode", type=str, choices=['cauchy', 'sigmoid', 'neg_cauchy'], default='cauchy')
        parser.add_argument("--sparse_loss_scale", type=float, default=4)
        parser.add_argument("--specular_weighted_sparse", action='store_true')
        parser.add_argument("--specular_sparse_scale", type=float, default=50)
        parser.add_argument("--alpha_bg_loss_weight", type=float, default=0)
        parser.add_argument("--use_transmission_mask", action='store_true')
        #visualization terms
        parser.add_argument("--return_normal_map", action='store_true')
        parser.add_argument("--return_agg_normal", action='store_true')
        parser.add_argument("--return_pred_normal", action='store_true')
        parser.add_argument(
            "--visual_items",
            type=str,
            nargs='*',
            default=None,
            help=
            "The (multiple) output items to show as images. This will replace the default visual items"
        )
        parser.add_argument(
            "--visual_items_additional",
            type=str,
            nargs='+',
            default=[],
            help=
            "The (multiple) output items to show as images in addition to default items. This is ignored if --visual_iterms is used"
        )
        parser.add_argument(
            '--out_channels',
            type=int,
            default=None,
            help=
            'number of output channels in decoder; default 4 for radiance, 8 for microfacet and others'
        )
        # ray generation
        parser.add_argument(
            '--which_ray_generation',
            type=str,
            default='cube',
            help='which ray point generation method to use [cube]')
        parser.add_argument('--domain_size',
                            type=int,
                            default=1,
                            help='Size of the ray marching domain')
        # rendering functions
        parser.add_argument('--which_render_func',
                            type=str,
                            default='microfacet',
                            help='which render method to use')
        parser.add_argument(
            '--which_blend_func',
            type=str,
            default='alpha',
            help=
            'which blend function to use. Hint: alpha2 for collocated, alpha for non-collocated'
        )
        parser.add_argument('--which_tonemap_func',
                            type=str,
                            default='gamma',
                            help='which tone map function to use.')


        parser.add_argument(
            '--num_pos_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for position encoding if using nerf or mixed mlp decoders'
        )
        parser.add_argument(
            '--num_viewdir_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for view direction encoding if using nerf decoders'
        )
        parser.add_argument('--num_normal_freqs', type=int, default=-1)
        parser.add_argument(
            '--num_feature_freqs',
            type=int,
            default=-1,
            help=
            'number of frequency for feature encoding if using mixed mlp decoders'
        )

        ### TODO add more...
        return parser

    def add_default_color_losses(self, opt):
        ''' if no color loss terms are specified, this function is called to 
            add default supervision into opt.color_loss_items
        '''

        opt.color_loss_items = []  # add this to actual names in subclasses

    def add_default_visual_items(self, opt):
        ''' if no visual terms are specified, this function is called to 
            add default visualization items
        '''
        opt.visual_items = ['gt_image'
                            ]  # add this to actual names in subclasses

    def check_setup_loss(self, opt):
        ''' this function check and setup all loss items and weights.'''

        self.loss_names = ['total']
        if not opt.color_loss_items:
            self.add_default_color_losses(opt)
        if len(opt.color_loss_weights) != 1 and len(
                opt.color_loss_weights) != len(opt.color_loss_items):
            print(fmt.RED + "color_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.color_loss_weights) == 1 and len(opt.color_loss_items) > 1:
            opt.color_loss_weights = np.ones(len(
                opt.color_loss_items), np.float32) * opt.color_loss_weights[0]
        self.loss_names += opt.color_loss_items

        if len(opt.depth_loss_weights) != 1 and len(
                opt.depth_loss_weights) != len(opt.depth_loss_items):
            print(fmt.RED + "color_depth_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.depth_loss_weights) == 1 and len(opt.depth_loss_items) > 1:
            opt.depth_loss_weights = np.ones(len(
                opt.depth_loss_items), np.float32) * opt.depth_loss_weights[0]
        self.loss_names += opt.depth_loss_items

        if len(opt.zero_one_loss_weights) != len(
                opt.zero_one_loss_items) and len(
                    opt.zero_one_loss_weights) != 1:
            print(fmt.RED + "zero_one_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.zero_one_loss_weights) == 1 and len(
                opt.zero_one_loss_items) > 1:
            opt.zero_one_loss_weights = np.ones(
                len(opt.zero_one_loss_items),
                np.float32) * opt.zero_one_loss_weights[0]
        self.loss_names += opt.zero_one_loss_items

        if len(opt.bg_loss_weights) != 1 and len(opt.bg_loss_weights) != len(
                opt.bg_loss_items):
            print(fmt.RED + "bg_loss_weights does not match loss items" +
                  fmt.END)
            exit()
        if len(opt.bg_loss_weights) == 1 and len(opt.bg_loss_items) > 1:
            opt.bg_loss_weights = np.ones(len(opt.bg_loss_items),
                                          np.float32) * opt.bg_loss_weights[0]
        self.loss_names += opt.bg_loss_items
        if opt.sparse_loss_weight > 0:
            self.loss_names += ["sparse"]
        if opt.eik_loss_weight > 0:
            self.loss_names += ["eikonal"]
        if opt.orientation_weight > 0:
            self.loss_names += ["orientation"]
        if opt.relsdf_loss_weight > 0:
            self.loss_names += ['relsdf_'+t for t in opt.relsdf_loss_items]
        if opt.point_normal_weight > 0:
            self.loss_names += ['point_normal']
        if opt.light_tv_weight > 0:
            self.loss_names += ['light_tv']
        if opt.light_achro_weight > 0:
            self.loss_names += ['light_achro']
        if opt.diffuse_tv_weight > 0:
            self.loss_names += ['diffuse_tv']
        if opt.roughness_tv_weight > 0:
            self.loss_names += ['roughness_tv']
        if opt.roughness_reg_weight > 0:
            self.loss_names += ['roughness_reg']
        if opt.normal_tv_weight > 0:
            self.loss_names += ['normal_tv']
        if opt.feat_tv_weight > 0:
            self.loss_names += ['feat_tv']
        if opt.sdf_tv_weight > 0:
            self.loss_names += ['sdf_tv']
        if opt.density_distill_weight > 0:
            self.loss_names += ['density_distill']
        if opt.light_reg_weight > 0:
            self.loss_names += ['light_reg']
        if opt.sdf_center_weight > 0:
            self.loss_names += ['sdf_center']
        if opt.vis_map_bce_weight > 0:
            self.loss_names += ['vis_map_bce']
        if opt.pnt_diffuse_weight > 0:
            self.loss_names += ['pnt_diffuse']
        if opt.mlp_diffuse_cor_weight > 0:
            self.loss_names += ['mlp_diffuse_cor']
        if opt.alpha_bg_loss_weight > 0:
            self.loss_names += ['alpha_bg']
        # add the functions used in losses
        self.l1loss = torch.nn.L1Loss().to(self.device)
        self.l2loss = torch.nn.MSELoss().to(self.device)
        self.color_loss = self.l2loss if opt.color_loss == 'l2' else self.l1loss

    def check_setup_visuals(self, opt):
        if opt.visual_items is None:
            print("visual_items not ", opt.visual_items)
            self.add_default_visual_items(opt)
            self.visual_names += opt.visual_items
            self.visual_names += opt.visual_items_additional
        else:
            self.visual_names += opt.visual_items

        if len(self.visual_names) == 0:
            print(fmt.YELLOW + "No items are visualized" + fmt.END)

    def create_network_models(self, opt):
        '''
        This function should create the rendering networks.
        Every subnetwork model needs to be named as self.net_"name",
        and the "name" needs to be added to the self.model_names list.
        An example of this is like:
            self.model_names = ['ray_marching']
            self.net_ray_marching = network_torch_model(self.opt)

            if self.opt.gpu_ids:
                self.net_ray_marching.to(self.device)
                self.net_ray_marching = torch.nn.DataParallel(
                    self.net_ray_marching, self.opt.gpu_ids)
        '''
        pass

    def run_network_models(self):
        '''
        This function defines how the network is run.
        This function should use the self.input as input to the network.
        and return a dict of output (that will be assign to self.output).
        If only a sinlge network is used, this function could be simply just:
            return net_module(**self.input)
        '''
        raise NotImplementedError()

    def prepare_network_parameters(self, opt):
        '''
        Setup the parameters the network is needed.
        By default, it finds rendering (shading) function, ray generation function, tonemap function, etc.
        '''

        self.check_setup_loss(opt)

        if len(self.loss_names) == 1 and opt.is_train == True:
            print(fmt.RED + "Requiring losses to train" + fmt.END)
            raise NotImplementedError()

        self.check_setup_visuals(opt)

        self.check_setup_renderFunc_channels(opt)

        self.blend_func = find_blend_function(opt.which_blend_func)
        self.raygen_func = find_ray_generation_method(opt.which_ray_generation)
        self.tonemap_func = find_tone_map(opt.which_tonemap_func)

        self.found_funcs = {}
        add_property2dict(
            self.found_funcs, self,
            ["blend_func", "raygen_func", "tonemap_func", "render_func"])

    def setup_optimizer(self, opt):
        '''
            Setup the optimizers for all networks.
            This assumes network modules have been added to self.model_names
            By default, it uses an adam optimizer for all parameters.
        '''

        params = []
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            params = params + list(net.parameters())

        self.optimizers = []

        self.optimizer = torch.optim.Adam(params,
                                          lr=opt.lr,
                                          betas=(0.9, 0.999))
        self.optimizers.append(self.optimizer)

    def check_opts(self, opt):
        pass

    def initialize(self, opt):
        super(BaseRenderingModel, self).initialize(opt)
        self.opt = opt

        if self.is_train:
            self.check_opts(opt)
        self.prepare_network_parameters(opt)

        self.create_network_models(opt)

        #check model creation
        if not self.model_names:
            print(
                fmt.RED +
                "No network is implemented! Or network's name is not properly added to self.model_names"
                + fmt.END)
            raise NotImplementedError()
        for mn in self.model_names:
            if not hasattr(self, "net_" + mn):
                print(fmt.RED + "Network " + mn + " is missing" + fmt.END)
                raise NotImplementedError()

        # setup optimizer
        if self.is_train:
            self.setup_optimizer(opt)

    def set_input(self, input):

        # setup self.input
        # this dict is supposed to be sent the network via **self.input in run_network_modules
        self.input = input
        for key, item in self.input.items():
            if isinstance(item, torch.Tensor):
                self.input[key] = item.to(self.device)

        # gt required in loss compute
        self.gt_image = self.input['gt_image'].to(
            self.device) if 'gt_image' in input else None

        self.gt_depth = self.input['gt_depth'].to(
            self.device) if 'gt_depth' in input else None

        self.gt_mask = self.input['gt_mask'].to(
            self.device) if 'gt_mask' in input else None
        

    def set_visuals(self):
        for key, item in self.output.items():
            if key in self.visual_names:
                setattr(self, key, item)
        if "coarse_raycolor" not in self.visual_names and "coarse_raycolor" in self.output:
            key = "coarse_raycolor"
            setattr(self, key, self.output[key])

    def check_setup_renderFunc_channels(self, opt):
        ''' Find render functions; 
            the function is often used by subclasses when creating rendering networks.
        '''

        self.render_func = find_render_function(opt.which_render_func)

        if opt.which_render_func == 'radiance':
            if opt.out_channels is None:
                opt.out_channels = 4
        elif opt.which_render_func == 'microfacet':
            if opt.out_channels is None:
                opt.out_channels = 8
        elif opt.which_render_func == 'harmonics':
            if opt.out_channels is None:
                opt.out_channels = 1 + 3 * 5 * 5
            deg = int(((opt.out_channels - 1) / 3)**0.5)
            if 1 + deg * deg * 3 != opt.out_channels:
                print(
                    fmt.RED +
                    '[Error] output channels should match the number of sh basis'
                    + fmt.END)
                exit()
            if deg <= 5:
                print("using SH table")
                self.shcomputer = SphericalHarm_table(deg)
            else:
                print("using runtime SH")
                self.shcomputer = SphericalHarm(deg)
            self.render_func.sphericalHarm = self.shcomputer
        else:
            if opt.out_channels is None:
                opt.out_channels = 8
        self.out_channels = opt.out_channels

    def check_getDecoder(self, opt, **kwargs):
        '''construct a decoder; this is often used by subclasses when creating networks.'''

        decoder = None
        if opt.which_decoder_model == 'mlp':
            decoder = MlpDecoder(num_freqs=opt.num_pos_freqs,
                                 out_channels=opt.out_channels,
                                 **kwargs)
        elif opt.which_decoder_model == 'viewmlp':
            decoder = ViewMlpDecoder(num_freqs=opt.num_pos_freqs,
                                     num_viewdir_freqs=opt.num_viewdir_freqs,
                                     num_channels=opt.out_channels,
                                     **kwargs)
        elif opt.which_decoder_model == 'viewmlpsml':
            decoder = ViewMlpSmlDecoder(num_freqs=opt.num_pos_freqs,
                                     num_viewdir_freqs=opt.num_viewdir_freqs,
                                     num_channels=opt.out_channels,
                                     **kwargs)
        elif opt.which_decoder_model == 'viewmlpmid':
            decoder = ViewMlpMidDecoder(num_freqs=opt.num_pos_freqs,
                                     num_viewdir_freqs=opt.num_viewdir_freqs,
                                     num_channels=opt.out_channels,
                                     **kwargs)
        elif opt.which_decoder_model == 'nv_mlp':
            decoder = VolumeDecoder(256,
                                    template_type=opt.nv_template_type,
                                    template_res=opt.nv_resolution,
                                    out_channels=opt.out_channels,
                                    **kwargs)
        elif opt.which_decoder_model == 'discrete_microfacet':
            decoder = DiscreteVolumeMicrofacetDecoder(
                opt.discrete_volume_folder,
                out_channels=opt.out_channels,
                **kwargs)
        elif opt.which_decoder_model == 'discrete_general':
            decoder = DiscreteVolumeGeneralDecoder(
                opt.discrete_volume_folder,
                out_channels=opt.out_channels,
                **kwargs)
        elif opt.which_decoder_model == 'mixed_mlp':
            decoder = MixedDecoder(256,
                                   template_type=opt.nv_template_type,
                                   template_res=opt.nv_resolution,
                                   mlp_channels=128,
                                   out_channels=opt.out_channels,
                                   position_freqs=opt.num_pos_freqs,
                                   feature_freqs=opt.num_feature_freqs,
                                   **kwargs)
        elif opt.which_decoder_model == 'mixed_separate_code':
            decoder = MixedSeparatedDecoder(
                256,
                template_type=opt.nv_template_type,
                template_res=opt.nv_resolution,
                mlp_channels=128,
                out_channels=opt.out_channels,
                position_freqs=opt.num_pos_freqs,
                feature_freqs=opt.num_feature_freqs,
                **kwargs)
        else:
            raise RuntimeError('Unknown decoder model: ' +
                               opt.which_decoder_model)

        return decoder



    def forward(self):
        self.output = self.run_network_models()

        self.set_visuals()

        if not self.opt.no_loss:
            self.compute_losses()

    def save_image(self, img_array, filepath):
        assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                             and img_array.shape[2] in [3, 4])

        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Image.fromarray(img_array).save(filepath)

    def compute_losses(self):
        ''' Compute loss functions.
            The total loss is saved in self.loss_total.
            Every loss will be set to an attr, self.loss_lossname
        '''

        self.loss_total = 0
        opt = self.opt
        #color losses
        # if opt.alpha_bg_loss_weight > 0 and 'alpha' in self.input:
        #     ray_mask = (self.input['alpha'] > 0).to(self.gt_image.device) * (self.output["ray_mask"][..., None] > 0)
        # else:
        ray_mask = self.output["ray_mask"][..., None] > 0
        if opt.use_transmission_mask:
            ray_mask = ray_mask * (self.output['coarse_is_background'] <= 0.8)

        for i, name in enumerate(opt.color_loss_items):
            if name.startswith("ray_masked_brdf"):
                unmasked_name = name[len("ray_masked")+1:]
                gt_name = name[len("ray_masked_brdf")+1:]
                
                masked_output = torch.masked_select(self.output[unmasked_name], ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                if 'combine' in gt_name:
                    masked_gt = torch.masked_select(self.gt_image, ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                else:
                    masked_gt = torch.masked_select(self.output[gt_name], ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3).detach()
                if masked_output.shape[1] > 0:
                    loss = self.color_loss(masked_output, masked_gt)
                else:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                # print("loss", name, torch.max(torch.abs(loss)))

            elif name.startswith("ray_masked"):
                unmasked_name = name[len("ray_masked")+1:]
                masked_output = torch.masked_select(self.output[unmasked_name], ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                masked_gt = torch.masked_select(self.gt_image, ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                if masked_output.shape[1] > 0:
                    loss = self.color_loss(masked_output, masked_gt)
                else:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                # print("loss", name, torch.max(torch.abs(loss)))

            elif name.startswith("ray_bg"):
                unmasked_name = name[len("ray_bg")+1:]
                masked_output = torch.masked_select(self.output[unmasked_name], ~ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                masked_gt = torch.masked_select(self.gt_image, ~ray_mask.expand(-1, -1, 3)).reshape(1, -1, 3)
                if masked_output.shape[1] > 0:
                    loss = self.color_loss(masked_output, masked_gt)
                else:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
                # print("loss", name, torch.max(torch.abs(loss)))
            
            elif name.startswith("ray_miss"):
                unmasked_name = name[len("ray_miss") + 1:]
                ## TODO add more ray_missed...
                if opt.low_trans_as_miss > 0:
                    ray_miss_mask = (self.output['coarse_is_background'][...,0] < opt.low_trans_as_miss) & self.output['ray_mask']
                else:
                    ray_miss_mask = self.output['ray_mask']
                masked_output = torch.masked_select(self.output[unmasked_name],
                                                    (ray_miss_mask == 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
                masked_gt = torch.masked_select(self.gt_image,(ray_miss_mask == 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)

                if masked_output.shape[1] > 0:
                    with torch.set_grad_enabled(opt.color_loss_weights[i]>0):
                        loss = self.l2loss(masked_output, masked_gt) * masked_gt.shape[1]
                else:
                    loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)

            elif name.startswith("ray_depth_masked"):
                pixel_xy = self.input["pixel_idx"][0].long()
                ray_depth_mask = self.output["ray_depth_mask"][0][pixel_xy[...,1], pixel_xy[...,0]] > 0
                unmasked_name = name[len("ray_depth_masked")+1:]
                masked_output = torch.masked_select(self.output[unmasked_name], (ray_depth_mask[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
                masked_gt = torch.masked_select(self.gt_image, (ray_depth_mask[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
                loss = self.l2loss(masked_output, masked_gt)

            else:
                if name not in self.output:
                    print(fmt.YELLOW + "No required color loss item: " + name +
                          fmt.END)
                # print("no_mask")
                with torch.set_grad_enabled(opt.color_loss_weights[i]>0):
                    loss = self.l2loss(self.output[name], self.gt_image)
                # print("loss", name, torch.max(torch.abs(loss)))
            self.loss_total += (loss * opt.color_loss_weights[i] + 1e-6)
            # loss.register_hook(lambda grad: print(torch.any(torch.isnan(grad)), grad, opt.color_loss_weights[i]))

            setattr(self, "loss_" + name, loss)
        # print(torch.sum(self.output["ray_mask"]))

        #depth losses
        for i, name in enumerate(opt.depth_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required depth loss item: " + name +
                      fmt.END)
            loss = self.l2loss(self.output[name] * self.gt_mask,
                               self.gt_depth * self.gt_mask)
            self.loss_total += loss * opt.depth_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        #background losses
        for i, name in enumerate(opt.bg_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required mask loss item: " + name +
                      fmt.END)
            if name in ['coarse_is_background', 'coarse_mask']:
                ray_mask = self.output["ray_mask"][..., None] > 0
                T = self.output[name][ray_mask].clamp(self.opt.zero_epsilon, 1 - self.opt.zero_epsilon)
                loss = (torch.log(T) + torch.log(1-T)).mean()
            else:
                loss = self.l2loss(self.output[name] * (1 - self.gt_mask),
                               1 - self.gt_mask)
            self.loss_total += loss * opt.bg_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        #zero_one regularization losses
        for i, name in enumerate(opt.zero_one_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required zero_one loss item: " + name +
                      fmt.END)
                # setattr(self, "loss_" + name, torch.zeros([1], device="cuda", dtype=torch.float32))
            else:
                val = torch.clamp(self.output[name], self.opt.zero_epsilon,
                                  1 - self.opt.zero_epsilon)
                # print("self.output[name]",torch.min(self.output[name]), torch.max(self.output[name]))
                loss = torch.mean(torch.log(val) + torch.log(1 - val))
                self.loss_total += loss * opt.zero_one_loss_weights[i]
                setattr(self, "loss_" + name, loss)

        # l2 square regularization losses
        for i, name in enumerate(opt.l2_size_loss_items):
            if name not in self.output:
                print(fmt.YELLOW + "No required l2_size_loss_item : " + name + fmt.END)
            loss = self.l2loss(self.output[name], torch.zeros_like(self.output[name]))
            # print("self.output[name]", self.output[name].shape, loss.shape)
            self.loss_total += loss * opt.l2_size_loss_weights[i]
            setattr(self, "loss_" + name, loss)

        if opt.alpha_bg_loss_weight > 0:
            if 'alpha' in self.input:
                alpha_bg_mask = (self.input['alpha'] <= 0.001).to(self.gt_image.device) * (self.output["ray_mask"][..., None] > 0)
                masked_output = torch.masked_select(self.output['coarse_is_background'], alpha_bg_mask).reshape(1, -1, 1)
                loss = ((1-masked_output)**2).mean()
            else:
                loss = 0
            self.loss_total += loss * opt.alpha_bg_loss_weight
            setattr(self, 'loss_alpha_bg', loss)

        if opt.sparse_loss_weight > 0:
            if opt.which_agg_model == 'sdfmlp':
                reg_density = self.output['reg_density']
                weight = self.output['dist_weight']
                # loss = (weight * torch.log(1 + 2*reg_sigma**2)).mean()
                # loss = (weight * (1-reg_sigma)**2).mean()
                if opt.specular_weighted_sparse:
                    specular_weight = self.output['specularity']
                    weight = weight * torch.exp(np.log(opt.specular_sparse_scale) * specular_weight)
                s = opt.sparse_loss_scale
                if opt.sparse_loss_mode == 'cauchy':
                    # loss = (weight * 0.5*torch.log1p(4*(1-reg_sigma)**2)).mean()
                    loss = (weight * 1/s*torch.log1p(s**2 *(1-reg_density)**2)).mean()
                elif opt.sparse_loss_mode == 'sigmoid':
                    loss = (weight * 4/s * torch.sigmoid(s*(0.5-reg_density))).mean()
                elif opt.sparse_loss_mode == 'neg_cauchy':
                    loss = (weight * 1/s*torch.log1p(s**2 *(reg_density)**2)).mean()
                # print(self.output['density'].shape, )
                self.loss_total += loss * opt.sparse_loss_weight
                setattr(self, "loss_sparse", loss)
            else:
                # weight and conf_coefficient 1, 1134, 40, 8
                if "weight" not in self.output or "conf_coefficient" not in self.output:
                    print(fmt.YELLOW + "No required sparse_loss_weight weight or conf_coefficient : " + fmt.END)

                loss = torch.sum(self.output["weight"] * torch.abs(1 - torch.exp(-2 * self.output["conf_coefficient"]))) / (torch.sum(self.output["weight"]) + 1e-6)
                # print("self.output[name]", self.output[name].shape, loss.shape)
                self.output.pop('weight')
                self.output.pop('conf_coefficient')
                self.loss_total += loss * opt.sparse_loss_weight
                setattr(self, "loss_sparse", loss)

        # self.loss_total = Variable(self.loss_total, requires_grad=True)
        
        grad_theta = None
        if opt.eik_loss_weight > 0 and 'agg_gradient' in self.output:
            weight = self.output['blend_weight'][...,0]
            mask = weight > 0.01
            weight = weight[mask]
            grad_theta = self.output['agg_gradient'][mask]
            grad_norm = grad_theta.norm(2, dim=-1)
            loss = (weight*(grad_norm - 1) ** 2).mean()
            self.loss_total += loss * opt.eik_loss_weight
            setattr(self, "loss_eikonal", loss)

        if opt.orientation_weight > 0 or opt.relsdf_loss_weight > 0 or opt.point_normal_weight > 0:
            if opt.point_normal_weight > 0 or not opt.learn_point_normal:
                if grad_theta is None:
                    grad_theta = self.output['agg_gradient']
                reg_grad = F.normalize(grad_theta, dim=-1)
                # tar_grad = reg_grad
            if opt.optim_compute_normal:
                tar_grad = reg_grad
            else:
                tar_grad = self.output['pred_gradient']

            if opt.orientation_weight > 0:
                dot_dir = 1 if opt.which_agg_model == 'sdfmlp' else -1
                weight = self.output['blend_weight'][...,0]
                mask = weight > 0.01
                weight = weight[mask] # 0.9 * weight[mask] + 0.1
                loss = (weight * 
                        torch.relu(dot_dir*(tar_grad[mask]*self.output['viewdirs'][mask]).sum(-1))**2).sum()
                self.loss_total += loss * opt.orientation_weight
                setattr(self, "loss_orientation", loss)

            if opt.relsdf_loss_weight > 0:
                if 'relsdf_mask' in self.output:
                    relsdf_weights = self.output['relsdf_weight'][...,:-1,0][self.output['relsdf_mask']]
                    # mask = relsdf_weights > 0.01
                    # relsdf_weights = relsdf_weights[mask] # 0.9 * relsdf_weights[mask] + 0.1
                    for name in opt.relsdf_loss_items:
                        loss = 0
                        if name == 'back_sdf':
                            mask = relsdf_weights > 0.1
                            mask = (self.output['sdf_d'] > 0) * mask
                            weights = relsdf_weights[mask] # (1-opt.relsdf_w_bias) * relsdf_weights[mask] + opt.relsdf_w_bias
                            # tan = self.output['sdf_d'][mask] / self.output['ray_dist_d'][mask].clamp_min(0.0005)
                            s_sq = self.output['sdf_d'][mask]**2
                            r_cos_sq = s_sq / (self.output['ray_dist_d'][mask].clamp_min(0.0005)**2  + s_sq)
                            loss = opt.relsdf_back_sdf_weight * (weights * r_cos_sq).sum() #  / self.output['ray_dist_d'][mask]
                            # loss = opt.relsdf_back_sdf_weight * (weights * torch.relu(self.output['sdf_d'][mask])**2).sum() #  / self.output['ray_dist_d'][mask]
                        if name == 'est_delta' and 'est_sdf_d' in self.output:
                            mask = (relsdf_weights > opt.relsdf_delta_thres)
                            if opt.relsdf_uni_weight:
                                weights = opt.relsdf_w_bias
                            else:
                                weights = (1-opt.relsdf_w_bias) * relsdf_weights[mask] + opt.relsdf_w_bias
                            orient_mask = self.output['sdf_d'][mask].detach() <= 0 if opt.relsdf_single_dir else 1
                            loss = (weights * orient_mask * (self.output['est_sdf_d'][mask] - self.output['sdf_d'][mask])**2).sum()
                        if name == 'dist_bound':
                            loss = (torch.relu(self.output['sdf_d'].abs() - self.output['ray_dist_d'])**2).sum()
                    
                        self.loss_total += loss * opt.relsdf_loss_weight
                        setattr(self, 'loss_relsdf_'+name, loss)

            if opt.point_normal_weight > 0:
                # + 0.1 to ensure every point normal have proper update.
                loss = ((0.9 * self.output['blend_weight'][...,0] + 0.1)* 
                        ((self.output['pred_gradient']-reg_grad.detach())**2).sum(-1)).sum()
                self.loss_total += loss * opt.point_normal_weight
                setattr(self, 'loss_point_normal', loss)

        if opt.diffuse_tv_weight > 0:
            loss = (self.output['blend_weight'] * self.output['pnt_diffuse_var']).sum()
            self.loss_total += loss * opt.diffuse_tv_weight
            setattr(self, 'loss_diffuse_tv', loss)

        if opt.roughness_tv_weight > 0:
            loss = (self.output['blend_weight'] * self.output['pnt_roughness_var']).sum()
            self.loss_total += loss * opt.roughness_tv_weight
            setattr(self, 'loss_roughness_tv', loss)

        if opt.roughness_reg_weight > 0:
            s = 4
            loss = (1/s*torch.log1p(s**2 *(self.output['roughness'] - opt.default_roughness)**2)).mean()
            self.loss_total += loss * opt.roughness_reg_weight
            setattr(self, 'loss_roughness_reg', loss)

        if opt.normal_tv_weight > 0:
            loss = (self.output['blend_weight'] * self.output['pnt_normal_var']).sum()
            self.loss_total += loss * opt.normal_tv_weight
            setattr(self, 'loss_normal_tv', loss)

        if opt.feat_tv_weight > 0:
            loss = (self.output['blend_weight'] * self.output['pnt_feat_var']).sum()
            self.loss_total += loss * opt.feat_tv_weight
            setattr(self, 'loss_feat_tv', loss)

        if opt.sdf_tv_weight > 0:
            loss = (self.output['blend_weight'] * self.output['pnt_sdf_var']).sum()
            self.loss_total += loss * opt.sdf_tv_weight
            setattr(self, 'loss_sdf_tv', loss)

        if opt.density_distill_weight > 0:
            # loss = ((self.output['reg_density_fixed'] - self.output['reg_density'])**2).mean()
            loss = F.binary_cross_entropy(self.output['reg_density'], self.output['reg_density_fixed'])
            self.loss_total += loss * opt.density_distill_weight
            setattr(self, 'loss_density_distill', loss)

        if opt.sdf_center_weight > 0:
            first_point_mask = self.output['ray_valid_order'] == 0
            loss = (torch.relu(-self.output['sdf'][first_point_mask])**2).sum()
            self.loss_total += loss * opt.sdf_center_weight
            setattr(self, 'loss_sdf_center', loss)

        if opt.pnt_diffuse_weight > 0:
            gt_key = 'diffuse_raycolor' if opt.learn_diffuse_color else 'coarse_raycolor'
            masked_output = torch.masked_select(self.output['point_raycolor'], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
            masked_gt = torch.masked_select(self.output[gt_key], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
            if masked_output.shape[1] > 0:
                loss = self.color_loss(masked_output, masked_gt.detach())
            else:
                loss = torch.tensor(0.0, dtype=torch.float32, device=masked_output.device)
            self.loss_total += loss * opt.pnt_diffuse_weight
            setattr(self, 'loss_pnt_diffuse', loss)

            # point_color = self.output['point_color']
            # diffuse_color = self.output['diffuse_color'].detach()
            # weights = self.output['blend_weight'].detach() * 0.9 + 0.1
            # loss = (weights * ((point_color - diffuse_color)**2).sum(-1, keepdim=True)).sum()
            # self.loss_total += loss * opt.pnt_diffuse_weight
            # setattr(self, 'loss_pnt_diffuse', loss)

        if opt.mlp_diffuse_cor_weight > 0:
            masked_r_diffuse = torch.masked_select(self.output['re_diffuse_raycolor'], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
            masked_p_diffuse = torch.masked_select(self.output['point_raycolor'], (self.output["ray_mask"] > 0)[..., None].expand(-1, -1, 3)).reshape(1, -1, 3)
            if masked_r_diffuse.shape[1] > 0:
                loss = ((masked_r_diffuse - masked_p_diffuse.detach())**2).mean()
            else:
                loss = torch.tensor(0.0, dtype=torch.float32, device=masked_r_diffuse.device)
            self.loss_total += loss * opt.mlp_diffuse_cor_weight
            setattr(self, 'loss_mlp_diffuse_cor', loss)

        if opt.brdf_rendering:
            if opt.light_tv_weight > 0:
                light = self.output['light_map']
                dx = light - torch.roll(light, 1, 1)
                dy = light - torch.roll(light, 1, 0)
                loss = torch.sum(dx ** 2 + dy ** 2)
                self.loss_total += opt.light_tv_weight * loss
                setattr(self, 'loss_light_tv', loss)
            if opt.light_achro_weight > 0:
                light = self.output['light_map']
                # dc = light - torch.roll(light, 1, 2)
                # loss = torch.sum(dc**2)
                # self.loss_total += opt.light_achro_weight * loss
                light_avg = light.mean(-1, keepdim=True)
                loss = ((light - light_avg)**2).mean(-1).sum()
                self.loss_total += opt.light_achro_weight * loss
                setattr(self, 'loss_light_achro', loss)
            if opt.light_reg_weight > 0:
                if opt.light_reg_mode == 'raw':
                    light = self.output['raw_light_map']
                    light_reg_thresh = np.log(opt.light_reg_thresh)
                elif opt.light_reg_mode == 'act':
                    light = self.output['light_map']
                    light_reg_thresh = opt.light_reg_thresh
                # s = 2
                # loss = (1/s*torch.log1p(s**2 *(opt.light_reg_thresh - light).relu()**2)).mean()
                loss = ((light_reg_thresh - light).relu()**2).mean()
                self.loss_total += opt.light_reg_weight * loss
                setattr(self, 'loss_light_reg', loss)

            if opt.vis_map_bce_weight > 0:
                pix_map = self.output['pixel_map'][...,:4,:]
                vis_map = self.output['vis_map'][...,None,:].expand_as(pix_map)
                loss = F.binary_cross_entropy(vis_map, pix_map)
                self.loss_total += opt.vis_map_bce_weight * loss
                setattr(self, 'loss_vis_map_bce', loss)

    def backward(self):
        self.optimizer.zero_grad()
        if self.opt.is_train:
            self.loss_total.backward()
            self.optimizer.step()

    def optimize_parameters(self, backward=True, total_steps=0):
        self.forward()
        self.backward()
