# encoding: utf-8
"""
@Author: yw
@Date: 2025/9/4 14:14

@Function:
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import utils.lr_sched as lr_sched
from data_loader.local_reader_vqvae_reconstruct import get_data
from rqvae_embed.rqvae_clip import RQVAE_EMBED_CLIP
from utils import dist_utils
from utils import logger
from utils.configs_utils import get_config
from utils.optim_factory import get_param_groups


def parse_arguments():
    """parse argument"""
    parser = argparse.ArgumentParser('RQVAE training', add_help=False)
    parser.add_argument('--cfg', default='configs/rqvae_i2v.yml', type=str)
    parser.add_argument('--output_dir', default='', help='Storage path')
    parser.add_argument("--tables", default='', help="ODPS table path")
    parser.add_argument('--resume', default='', help='Recovery checkpoint path')
    parser.add_argument('--train_root', default='', help='Training data root directory')
    parser.add_argument('--epochs', default=0, type=int, help='Number of training epochs')

    # 设备与分布式：在前面指定 device_type，后续代码设备无关
    parser.add_argument('--device_type', default='', type=str, choices=['cuda', 'npu', ''],
                        help='Device: cuda or npu. Default from config.')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help="Local ranking in distributed training. Automatically imported by PAI or XDL launcher")
    parser.add_argument('--dist_url', default='env://', help='Set the URL for distributed training')
    parser.add_argument('--distributed', action='store_true', help='Whether to enable distributed training')
    parser.add_argument('--save_prefix', default='test', help="Save Prefix")

    return parser.parse_args()


def gather_tensors(tensor):
    """gather tensors on all processes"""
    world_size = dist.get_world_size()
    with torch.no_grad():
        tensors_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_list, tensor)
    gathered_tensor = torch.cat(tensors_list, dim=0)
    return gathered_tensor


def initialize_training_environment(args, config):
    """init environment"""
    dist_utils.init_distributed_mode(config, args)
    dist_utils.main_pprint(OmegaConf.to_object(config))


def create_model(config):
    """create model"""
    hps = {
        "bottleneck_type": "rq",
        "embed_dim": config.model.codebook_dim,
        "n_embed": config.model.codebook_size,
        "latent_shape": [8, 8, config.model.codebook_dim],
        "code_shape": [8, 8, config.model.codebook_num],
        "shared_codebook": config.model.shared_codebook,
        "decay": config.model.decay,
        "restart_unused_codes": config.model.restart_unused_codes,
        "loss_type": config.model.loss_type,
        "latent_loss_weight": config.model.latent_loss_weight,
        "masked_dropout": 0.0,
        "use_padding_idx": False,
        "VQ_ema": config.model.VQ_ema,
        "latent_weight": eval(config.model.latent_weight),
        'do_bn': config.model.do_bn,
        'rotation_trick': config.model.rotation_trick
    }

    ddconfig = {
        "double_z": False,
        "z_channels": config.model.codebook_dim,
        "input_dim": config.model.input_dim,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [8],
        "dropout": 0.00
    }

    model_instance = RQVAE_EMBED_CLIP(hps=hps, ddconfig=ddconfig, checkpointing=True)

    return model_instance


def _to_numpy_array(x):
    """统一转换为numpy数组"""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def compute_codebook_utilization(codes_list, codebook_sizes):
    """
    计算每层码本的利用率（使用的code数量 / 总code数量）
    从实际训练数据中统计每一层码本的使用情况
    
    Args:
        codes_list: list, code数组的列表，每个code shape: [h, w, codebook_num]
        codebook_sizes: list, 每层码本的大小列表，如 [8192, 8192, 8192]
    
    Returns:
        dict: 包含每层利用率的字典
    """
    if not codes_list:
        return {}
    
    num_layers = len(codebook_sizes)
    utilization_stats = {}
    
    # 统一转换为numpy数组，避免重复转换
    codes_np = [_to_numpy_array(code) for code in codes_list]
    
    # 对每一层计算利用率
    for layer_idx in range(num_layers):
        used_codes_set = set()
        
        for code in codes_np:
            # 提取第layer_idx层的code: [h, w, codebook_num] -> [h, w]
            code_layer = code[:, :, layer_idx]
            # 统计unique codes
            used_codes_set.update(code_layer.flatten().tolist())
        
        total_codes = codebook_sizes[layer_idx]
        used_codes = len(used_codes_set)
        utilization = used_codes / total_codes if total_codes > 0 else 0.0
        
        utilization_stats[f'codebook_{layer_idx}_utilization'] = utilization
        utilization_stats[f'codebook_{layer_idx}_used_codes'] = used_codes
        utilization_stats[f'codebook_{layer_idx}_total_codes'] = total_codes
    
    return utilization_stats


def compute_non_collision_rate(codes_list, sample_ratio=1.0):
    """
    计算商品非冲突率（unique code数量 / 总商品数）
    
    定义：一个商品用所有层码本表示为一个完整的code，非冲突率 = unique code数量 / 总商品数
    - 如果所有商品都有不同的code，非冲突率 = 1.0（100%，无冲突）
    - 如果有商品共享相同的code，非冲突率 < 1.0（有冲突）
    
    Args:
        codes_list: list, code数组的列表，每个code shape: [h, w, codebook_num]，包含所有层的码本
        sample_ratio: float, 采样比例，用于减少计算负担（默认1.0表示全部统计）
    
    Returns:
        dict: 包含非冲突率的字典
    """
    if not codes_list:
        return {}
    
    # 采样以减少计算负担
    if sample_ratio < 1.0:
        import random
        sample_size = max(1, int(len(codes_list) * sample_ratio))
        codes_list = random.sample(codes_list, sample_size)
    
    # 统一转换为numpy数组并转换为tuple（可哈希）
    code_to_count = {}
    for code in codes_list:
        code_np = _to_numpy_array(code)
        code_tuple = tuple(code_np.flatten().tolist())
        code_to_count[code_tuple] = code_to_count.get(code_tuple, 0) + 1
    
    num_items = len(codes_list)
    unique_code_count = len(code_to_count)
    non_collision_rate = unique_code_count / num_items if num_items > 0 else 0.0
    collided_items = sum(count - 1 for count in code_to_count.values() if count > 1)
    
    return {
        'non_collision_rate': non_collision_rate,
        'unique_codes': unique_code_count,
        'total_items': num_items,
        'collided_items': collided_items,
    }


def prepare_optimizer_and_scheduler(config, model_without_ddp):
    """Prepare the optimizer and learning rate scheduler"""
    effective_batch_size = config.data.batch_size * config.train.accum_iter * dist_utils.get_world_size()

    # config.output_dir += f"{config.data.save_prefix}_ebs{effective_batch_size}_lr{config.train.lr}_ep{config.train.epochs}"

    current_time = int(time.time())
    formatted_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')

    # config.output_dir += f'_{formatted_time}'

    if config.train.lr is None:
        config.train.lr = config.train.blr * effective_batch_size / 256

    dist_utils.main_print(f"base lr: {round(config.train.lr * 256 / effective_batch_size, 6)}")
    dist_utils.main_print(f"lr: {round(config.train.lr, 6)}")
    dist_utils.main_print(f"Gradient accumulation: {config.train.accum_iter}")
    dist_utils.main_print(f"Effective batch size: {effective_batch_size}")

    param_groups = get_param_groups(config, model_without_ddp)

    optimizer_instance = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.optimizer.betas)

    return optimizer_instance


def main():
    """main entrance"""
    args = parse_arguments()
    cfg = get_config(args)

    initialize_training_environment(args=args, config=cfg)

    # 设备在入口统一指定，后续全部使用 device，与 cuda/npu 解耦
    device = dist_utils.get_device(cfg)

    seed = cfg.seed + dist_utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    elif device.type == 'npu':
        torch.npu.manual_seed(seed)

    OmegaConf.set_readonly(cfg, False)
    current_timestamp_seconds = int(time.time())
    formatted_timestamp = datetime.datetime.fromtimestamp(current_timestamp_seconds).strftime('%Y%m%d_%H%M%S')
    cfg.output_dir += f"{cfg.data.save_prefix}_bs{cfg.data.batch_size}_lr{cfg.train.lr}_ep{cfg.train.epochs}_{formatted_timestamp}"

    if dist_utils.get_rank() == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(os.path.join(cfg.output_dir, 'config.json'), 'w') as file_handle:
            json.dump(config_dict, file_handle, indent=4)

    # -----------------------------------------------------------------------------------------------
    # Build model
    dist_utils.main_print("building model...")
    model = create_model(cfg)

    # -----------------------------------------------------------------------------------------------
    # Resume training
    OmegaConf.set_readonly(cfg, False)
    if cfg.resume:
        state_dict = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        cfg.train.start_epoch = state_dict['epoch'] + 1

    model.to(device)

    model_without_ddp = model
    number_of_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dist_utils.main_print(f"Model: {str(model_without_ddp)}")
    dist_utils.main_print(f"Parametes of Model(M): {number_of_trainable_params / 1e6:.2f}")

    if cfg.dist.distributed:
        dist_utils.main_print("Model distributed data parallelism")
        # device_ids 使用本地设备 ID（LOCAL_RANK），对于 NPU 和 CUDA 都适用
        # cfg.dist.gpu 在 init_distributed_mode 中已设置为 LOCAL_RANK
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cfg.dist.gpu],  # LOCAL_RANK，节点内的设备 ID
            find_unused_parameters=True
        )
        # 保证 device 指向 DDP 所在设备
        device = next(model.parameters()).device

        model_without_ddp = model.module

    # -----------------------------------------------------------------------------------------------
    # Initialize dataset and dataloader
    dist_utils.main_print("Creating dataset and data loader...")
    # 检查实际使用的数据源：train_root（npz 文件）或 tables（ODPS 表，未实现）
    assert cfg.data.train_root or len(cfg.data.tables) > 0, 'No Data! Please specify train_root or tables.'
    data = get_data(cfg=cfg, epoch_id=0)
    for key in data:
        dist_utils.main_print(f"The size of the training dataset {key}: {len(data[key].dataset)}")

    # -----------------------------------------------------------------------------------------------
    # Initialize optimizer and lr scheduler
    dist_utils.main_print("Creating optimizer and learning rate scheduler...")
    optimizer = prepare_optimizer_and_scheduler(config=cfg, model_without_ddp=model_without_ddp)
    dist_utils.main_print(optimizer)
    OmegaConf.set_readonly(cfg, True)

    # -----------------------------------------------------------------------------------------------
    # training
    dist_utils.main_print(f"start training {cfg.train.epochs} epoch...")
    dist_utils.main_print(f"output dir: {cfg.output_dir}")
    start_time = time.time()

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(model, data, optimizer, epoch, cfg=cfg, device=device)
        if cfg.output_dir and (epoch % 1 == 0 or epoch + 1 == cfg.train.epochs):
            # 只在主进程保存checkpoint，避免多进程同时写入导致的问题
            if dist_utils.is_main_process():
                os.makedirs(cfg.output_dir, exist_ok=True)  # 确保目录存在
                checkpoint_save_info = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'cfg': OmegaConf.to_container(cfg),
                }
                torch.save(checkpoint_save_info, os.path.join(cfg.output_dir, f'checkpoint-{epoch}.pth'))

        dist_utils.main_print('*' * 100)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': number_of_trainable_params}

        if cfg.output_dir and dist_utils.is_main_process():
            os.makedirs(cfg.output_dir, exist_ok=True)
            with open(os.path.join(cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    dist_utils.main_print(f"Training Time: {total_time_str}")


def train_one_epoch(model: torch.nn.Module, data: dict, optimizer: torch.optim.Optimizer,
                    epoch: int, cfg=None, device=None):
    """
    Executes a training epoch.
    device: 由入口 get_device(cfg) 传入，本函数内仅使用 .to(device)，与 cuda/npu 解耦。
    """
    if device is None:
        device = next(model.parameters()).device

    metric_logger = logger.MetricLogger(delimiter="  ", device=device)
    header = f'Epoch: [{epoch}]'
    print_freq = cfg.log_step

    model.train(True)
    accum_iter = cfg.train.accum_iter
    
    # 用于统计商品冲突率和码本利用率
    # 统计频率从config读取，默认每10个epoch统计一次
    eval_cfg = getattr(cfg, 'eval', None)
    stats_freq = getattr(eval_cfg, 'stats_freq', 10) if eval_cfg else 10  # 每N个epoch统计一次
    collision_sample_ratio = getattr(eval_cfg, 'collision_sample_ratio', 0.1) if eval_cfg else 0.1
    collect_stats = (stats_freq > 0 and (epoch % stats_freq == 0 or epoch == cfg.train.epochs - 1))

    dataloader, sampler = data['recon'].dataloader, data['recon'].sampler
    data_iter = iter(dataloader)
    if sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch_list = [dataloader.num_batches]
    data_iter_list = [data_iter]
    dataset_names = ['recon']

    for key in data:
        if key != 'recon' and hasattr(data[key], 'dataloader') and hasattr(data[key], 'sampler'):
            dataloader = data[key].dataloader
            sampler = data[key].sampler
            if sampler is not None:
                sampler.set_epoch(epoch)

            num_batches_per_epoch_list.append(dataloader.num_batches)
            data_iter = iter(dataloader)
            data_iter_list.append(data_iter)
            dataset_names.append(key)

    num_batches_per_epoch = sum(num_batches_per_epoch_list)
    optimizer.zero_grad()

    # 开始训练循环
    dist_utils.main_print('=======>')
    dist_utils.main_print(f'Start {epoch}')

    for data_iter_step, (batch, select_idx, dataset_name) in enumerate(
            metric_logger.log_every_list_with_datasetname(data_iter_list, num_batches_per_epoch_list, dataset_names,
                                                          print_freq, epoch,
                                                          header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_batches_per_epoch + epoch, cfg)

        if select_idx == 0:
            # recon数据集：返回 (item_id, embedding)
            item_ids, features = batch
            features = features.to(device, non_blocking=True)
            loss, recons, selected_index, loss_dict, feature_norm, quant_norm = model(features)
            

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            lr2 = optimizer.param_groups[-1]["lr"]

            dist_utils.device_synchronize(device)

            log_metrics = {
                f"loss/loss": loss_value,
                f"loss/recon_loss": loss_dict['recon_loss'].item(),
                f"loss/cmt_loss": loss_dict['commitment_loss'].item(),
                "lr": lr,
                "lr2": lr2,
            }
            metric_logger.update(**log_metrics)

        # 对比
        else:
            _, features, _, tar_features = batch
            features = features.to(device, non_blocking=True)
            tar_features = tar_features.to(device, non_blocking=True)
            output = model(features, tar_features, return_clip_loss=True)
            loss = output['loss']

            # commitment loss
            clip_loss_weight = 1.
            loss = loss * clip_loss_weight + output['commitment_loss']

            loss_ori = output['loss_ori'].item()
            loss_self = output['loss_self'].item()
            loss_cl = output['loss_cl'].item()

            # 计算准确率
            acc = output['clip_acc'].item()

            model_unwrapped = dist_utils.get_model(model)
            temperature = model_unwrapped.logit_scale.item()
            temperature_self = model_unwrapped.logit_scale_self.item()
            temperature_cl = model_unwrapped.logit_scale_cl.item()
            loss_value = loss.item()

            commitment_loss = output['commitment_loss'].item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dist_utils.device_synchronize(device)

            lr = optimizer.param_groups[0]["lr"]

            log_metrics = {
                f"loss/loss": loss_value,
                f"loss/loss_ori": loss_ori,
                f"loss/loss_self": loss_self,
                f"loss/loss_cl": loss_cl,
                f"loss/cmt_loss": commitment_loss,
                f"acc": acc,

                "lr": lr,
                f"temp/temperature": temperature,
                f"temp/temperature_self": temperature_self,
                f"temp/temperature_cl": temperature_cl,
            }
            metric_logger.update(**log_metrics)

    metric_logger.synchronize_between_processes(device=device)
    dist_utils.main_print("Averaged stats:", metric_logger)
    
    # 评估：计算码本利用率和商品冲突率
    # 使用evaluate函数，支持DDP模式，在epoch训练结束后用最终权重评估
    eval_cfg = getattr(cfg, 'eval', None)
    stats_freq = getattr(eval_cfg, 'stats_freq', 10) if eval_cfg else 10  # 每N个epoch评估一次
    if epoch % stats_freq == 0 or epoch == cfg.train.epochs - 1:
        dist_utils.main_print("Evaluating codebook utilization and collision rate...")
        eval_stats = evaluate(model, data, cfg, device)
        
        # 打印评估结果（只在主进程打印）
        if dist_utils.is_main_process() and eval_stats:
            # 打印码本利用率
            num_codebooks = len([k for k in eval_stats.keys() if 'utilization' in k])
            for i in range(num_codebooks):
                used = int(eval_stats.get(f'codebook_{i}_used_codes', 0))
                total = int(eval_stats.get(f'codebook_{i}_total_codes', 0))
                utilization = eval_stats.get(f'codebook_{i}_utilization', 0.0)
                dist_utils.main_print(f"codebook_{i}_utilization: {used}/{total}={utilization:.4f} ({utilization*100:.2f}%)")
            
            # 打印非冲突率
            non_collision_rate = eval_stats.get('non_collision_rate', 0.0)
            unique_codes = int(eval_stats.get('unique_codes', 0))
            total_items = int(eval_stats.get('total_items', 0))
            collided_items = int(eval_stats.get('collided_items', 0))
            dist_utils.main_print(f"non_collision_rate: {unique_codes}/{total_items}={non_collision_rate:.4f} ({non_collision_rate*100:.2f}%), collided_items={collided_items}")
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, data, cfg, device):
    """
    评估模型：计算码本利用率和商品非冲突率
    支持DDP模式，每个进程处理部分数据，最后在rank 0汇总结果
    
    Args:
        model: 模型实例
        data: 数据字典，包含'recon'数据集
        cfg: 配置对象
        device: 设备
    
    Returns:
        dict: 包含利用率和冲突率的统计字典（只在rank 0返回结果，其他进程返回空dict）
    """
    model.eval()
    
    # 获取评估batch size（使用valid_batch_size）
    eval_cfg = getattr(cfg, 'eval', None)
    eval_batch_size = getattr(eval_cfg, 'valid_batch_size', cfg.data.batch_size) if eval_cfg else cfg.data.batch_size
    collision_sample_ratio = getattr(eval_cfg, 'collision_sample_ratio', 0.1) if eval_cfg else 0.1
    
    # 创建评估用的dataloader（使用valid_batch_size）
    recon_dataset = data['recon'].dataset
    recon_sampler = torch.utils.data.distributed.DistributedSampler(
        recon_dataset, 
        shuffle=False  # 评估时不需要shuffle
    )
    eval_dataloader = torch.utils.data.DataLoader(
        recon_dataset,
        batch_size=eval_batch_size,
        sampler=recon_sampler,
        pin_memory=True,
        num_workers=1,  # 评估时减少workers
        drop_last=False  # 评估时不drop last batch
    )
    
    codes_list = []
    
    # 每个进程处理自己的数据
    with torch.no_grad():
        # 只在主进程显示进度
        if dist_utils.is_main_process():
            from tqdm import tqdm
            eval_iter = tqdm(eval_dataloader, desc='Evaluating', leave=False)
        else:
            eval_iter = eval_dataloader
        
        for batch_idx, (item_ids, features) in enumerate(eval_iter):
            features = features.to(device, non_blocking=True)
            
            # 获取code（使用当前的模型权重）
            model_unwrapped = dist_utils.get_model(model)
            codes = model_unwrapped.rq_model.get_codes(features)
            # codes shape: [batch_size, h, w, codebook_num]
            codes_cpu = codes.cpu().numpy()
            # 按batch维度拆分，每个元素shape: [h, w, codebook_num]
            for i in range(codes_cpu.shape[0]):
                codes_list.append(codes_cpu[i])
    
    # DDP模式下，收集所有进程的codes_list到rank 0
    if dist_utils.is_dist_avail_and_initialized():
        codes_array = np.array(codes_list)  # shape: [N, h, w, codebook_num]
        codes_tensor = torch.from_numpy(codes_array).to(device)
        gathered_codes = [torch.zeros_like(codes_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_codes, codes_tensor)
        
        # 在rank 0汇总
        if dist.get_rank() == 0:
            all_codes_list = []
            for codes in gathered_codes:
                codes_np = codes.cpu().numpy()
                if len(codes_np) > 0:
                    for i in range(len(codes_np)):
                        all_codes_list.append(codes_np[i])
            codes_list = all_codes_list
        else:
            codes_list = []
    
    # 计算统计结果（只在主进程计算）
    stats = {}
    if dist_utils.is_main_process():
        # 获取每层码本的大小
        model_unwrapped = dist_utils.get_model(model)
        quantizer = model_unwrapped.rq_model.quantizer
        codebook_sizes = [codebook.n_embed for codebook in quantizer.codebooks]
        
        # 码本利用率：使用所有数据
        utilization_stats = compute_codebook_utilization(codes_list, codebook_sizes)
        stats.update(utilization_stats)
        
        # 商品非冲突率：可采样
        non_collision_stats = compute_non_collision_rate(codes_list, sample_ratio=collision_sample_ratio)
        stats.update(non_collision_stats)
    
    model.train()  # 恢复训练模式
    return stats


if __name__ == '__main__':
    main()
