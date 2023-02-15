import configargparse
import os
from models import find_model_class_by_name
from data import find_dataset_class_by_name
import torch


class BaseOptions:
    def initialize(self, parser: configargparse.ArgumentParser):
        #================================ global ================================#
        parser.add_argument('--base_config', is_config_file=True, 
                        help='config file path')
        parser.add_argument('--spidr_config', is_config_file=True, 
                        help='config file path')
        parser.add_argument('--config', is_config_file=True)
        # Note on priority of three configs
        # base_config < config < spidr_config
        # ==== spidr related ====
        parser.add_argument('--sdf_config', type=str, default='../dev_scripts/spidr/sdf_config.ini')
        parser.add_argument('--lighting_config', type=str, default='../dev_scripts/spidr/lighting_config.ini')
        parser.add_argument('--run_mode', type=str, default='pointnerf', choices=['sdf', 'lighting', 'pointnerf'])

        parser.add_argument('--name',
                            type=str,
                            default='exp',
                            help='name of the experiment')
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='if specified, print more debugging information')
        parser.add_argument(
            '--timestamp',
            action='store_true',
            help='suffix the experiment name with current timestamp')
        #================================ dataset ================================#
        parser.add_argument('--data_root',
                            type=str,
                            default=None,
                            help='path to the dataset storage')
        parser.add_argument(
            '--dataset_name',
            type=str,
            default=None,
            help='name of dataset, determine which dataset class to use')
        parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=float("inf"),
            help='Maximum number of samples allowed per dataset.'
            'If the dataset directory contains more than max_dataset_size, only a subset is loaded.'
        )
        parser.add_argument('--n_threads',
                            default=1,
                            type=int,
                            help='# threads for loading data')
        #================================ MVS ================================#

        parser.add_argument('--geo_cnsst_num',
                            default=2,
                            type=int,
                            help='# threads for loading data')


        #================================ model ================================#
        parser.add_argument('--bgmodel',
                            default="No",
                            type=str,
                            help='No | sphere | plane')

        parser.add_argument(
            '--model',
            type=str,
            default='mvs_points_volumetric',
            help='name of model, determine which network model to use')

        #================================ running ================================#
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='input batch size')
        parser.add_argument('--render_only',
                            type=int,
                            default=0,
                            help='1 for render_only dataset')
        parser.add_argument('--serial_batches',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
        parser.add_argument('--gpu_ids',
                            type=str,
                            default='0',
                            help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir',
                            type=str,
                            default='./checkpoints',
                            help='models are saved here')
        parser.add_argument('--show_tensorboard', action='store_true',
                            help='plot loss curves with tensorboard')
        parser.add_argument('--resume_dir',
                            type=str,
                            default='',
                            help='dir of the previous checkpoint')
        parser.add_argument('--resume_iter',
                            type=str,
                            default='latest',
                            help='which epoch to resume from')
        parser.add_argument('--debug',
                            action='store_true',
                            help='indicate a debug run')
        parser.add_argument('--vid',
                            type=int,
                            default=0,
                            help='feed batches in order without shuffling')
        parser.add_argument('--resample_pnts',
                            type=int,
                            default=-1,
                            help='resample the num. initial points')
        parser.add_argument('--inall_img',
                            type=int,
                            default=1,
                            help='all points must in the sight of all camera pose')
        parser.add_argument('--test_train', type=int, default=0, help='test on training set for debugging')
        parser.add_argument('--resume_points_only', action='store_true')
        parser.add_argument('--test_start_id', type=int, default=0)
        parser.add_argument('--test_end_id', type=int, default=-1)
        parser.add_argument('--light_save_id', type=int, default=-1)
        parser.add_argument('--resume_fixed_iter', type=str, default='')
        parser.add_argument('--test_ids', type=int, nargs='+', default=[])
        return parser

    def gather_options(self):
        parser = configargparse.ArgumentParser(
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt, _ = self.cfg_parse(parser)

        model_name = opt.model
        find_model_class_by_name(model_name).modify_commandline_options(
            parser, self.is_train)

        dataset_name = opt.dataset_name
        if dataset_name is not None:
            find_dataset_class_by_name(
                dataset_name).modify_commandline_options(
                    parser, self.is_train)

        self.parser = parser

        opt, unknown = self.cfg_parse(parser)
        assert len(unknown) == 0, 'unknown options: {}'.format(unknown)
        return opt

    def print_and_save_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # if opt.is_train:
        #     expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # else:
        #     expr_dir = os.path.join(opt.resume_dir, opt.name)
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def cfg_parse(self, parser):
        opt, _ = parser.parse_known_args()
        cfg_parse = ''
        cfg_parse = cfg_parse + f'--base_config {opt.base_config} ' if opt.base_config is not None else cfg_parse
        if opt.run_mode == 'sdf' and opt.sdf_config is not None:
            opt.spidr_config = opt.sdf_config
        elif opt.run_mode == 'lighting' and opt.lighting_config is not None:
            opt.spidr_config = opt.lighting_config
        cfg_parse = cfg_parse + f'--spidr_config {opt.spidr_config} ' if opt.spidr_config is not None else cfg_parse
        cfg_parse = cfg_parse + f'--config {opt.config} ' if opt.config is not None else cfg_parse
        opt, _ = parser.parse_known_args(cfg_parse)
        return parser.parse_known_args(namespace=opt)

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train

        if opt.timestamp:
            import datetime
            now = datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S')
            opt.name = opt.name + '_' + now

        self.print_and_save_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [
            int(x) for x in opt.gpu_ids.split(',') if x.strip() and int(x) >= 0
        ]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
