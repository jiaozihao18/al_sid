"""
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 --master_port=27646 kmeans.py

"""

from typing import Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch import einsum

from utils import dist_utils


def noop(*args, **kwargs):
    pass


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min=0).sqrt()


def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)


def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype=dtype, device=device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target


def kmeans(
        samples,
        num_clusters,
        num_iters=10,
        use_cosine_sim=False,
        sample_fn=batched_sample_vectors,
        all_reduce_fn=noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim=-1)
        bins = batched_bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype=dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d=dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins, buckets


def residual_kmeans(samples, num_clusters: Union[list, torch.Tensor], num_iters=100):
    # Initial KMeans clustering
    res_centers = []
    for num_cluster in num_clusters:
        centers, _, buckets = kmeans(samples, num_cluster, num_iters)
        res_centers.append(centers.squeeze())
        samples = samples - centers[:, buckets.squeeze()]
    # if dist_utils.is_dist_avail_and_initialized():
    #     res_centers = dist_utils.scaled_all_reduce(res_centers)
    return res_centers


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--cfg', default='configs/rqvae_i2v.yml', type=str)
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='', help='path where to save on oss')
    parser.add_argument('--resume', default='', help='Resumed checkpoint path')
    parser.add_argument('--train_root', default='', help='training data')
    parser.add_argument('--epochs', default=0, type=int, help='training epochs')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int, help="")
    parser.add_argument('--gpu', default=0, type=int, help="")
    parser.add_argument('--local-rank', default=-1, type=int,
                        help="For distributed training: local_rank. Automatically input in from PAI or XDL launcher")
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true', help='distributed')
    parser.add_argument('--save_prefix', default='test', help="保存路径")

    dist_utils.init_distributed_mode2(cfg, args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    x = torch.rand(1, 1000, 64)
    means = residual_kmeans(x, [128, 125, 128])
    for mean in means:
        print(mean.shape)
