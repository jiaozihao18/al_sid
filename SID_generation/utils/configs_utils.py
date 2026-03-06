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


def get_config(config_path):
    """
    从 yml 路径加载配置，不做命令行覆盖。
    config_path: 配置文件路径，如 'configs/rqvae_i2v.yml'
    """
    cfg = load_config(config_path)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    return cfg
