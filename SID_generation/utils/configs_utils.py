# encoding: utf-8
"""
@author: Yingwu.XSW
@date: 2022/3/24 下午5:07
"""

import argparse
import os.path as osp
import time
from datetime import datetime
from importlib import import_module

from omegaconf import OmegaConf


def load_config_and_rewrite(args):
    """
        load configs and change the value in namespace
    """
    mod = import_module(args.extra_config)
    mod = mod.cfg

    args = vars(args)
    for k, v in mod.items():
        args[k] = v
    args = argparse.Namespace(**args)
    return args


def merge_b_to_a(args_a, args_b):
    """
        load configs and change the value in namespace
    """
    args_a = vars(args_a)
    args_b = vars(args_b)

    for k, v in args_b.items():
        if k not in args_a:
            args_a[k] = v
    args = argparse.Namespace(**args_a)
    return args


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.abspath(osp.join(osp.dirname(cfg_file), cfg._base_)))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def get_config(args):
    cfg = load_config(args.cfg)

    # cfg.tmp_path = f'tmp_{int(time.time())}'

    if hasattr(args, 'valid_batch_size') and args.valid_batch_size > 0:
        if not hasattr(cfg, 'eval'):
            from omegaconf import DictConfig
            cfg.eval = DictConfig({})
        cfg.eval.valid_batch_size = args.valid_batch_size

    if hasattr(args, 'output_dir') and len(args.output_dir) > 0:
        cfg.output_dir = args.output_dir

    if hasattr(args, 'resume') and len(args.resume) > 0:
        cfg.resume = args.resume

    if hasattr(args, 'train_root') and len(args.train_root) > 0:
        cfg.data.train_root = args.train_root

    if hasattr(args, 'epochs') and args.epochs > 0:
        cfg.train.epochs = args.epochs

    if hasattr(args, 'LR') and args.LR > 0:
        cfg.train.lr = args.LR

    if hasattr(args, 'output_table') and len(args.output_table) > 0:
        cfg.output_table = args.output_table

    if hasattr(args, 'save_prefix') and len(args.save_prefix) > 0:
        cfg.data.save_prefix = args.save_prefix

    if hasattr(args, 'device_type') and str(args.device_type).strip() != '':
        cfg.dist.device_type = str(args.device_type).strip().lower()

    if hasattr(args, 'input_dim') and len(args.input_dim) > 0:
        cfg.model.input_dim = args.input_dim
        
    # table (ODPS 表功能未实现，保留此逻辑以兼容可能的未来扩展)
    if hasattr(args, 'tables') and args.tables:
        cfg.data.tables = args.tables
        cfg.data.train_data = args.tables
        cfg.data.val_data = ''

    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)

    return cfg
