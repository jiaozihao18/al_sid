import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import dist_utils
from .rqvae import RQVAE_EMBED


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# 定义余弦距离损失
def cosine_loss(output, target):
    cosine_sim = F.cosine_similarity(output, target, dim=-1)
    loss = 1 - cosine_sim.mean()
    return loss


def l2norm(t, dim=-1, eps=1e-6):
    return F.normalize(t, p=2, dim=dim, eps=eps)


# sampling helpers
def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def cal_pos_neg(image_embed, text_embed, local_batch_size):
    logits_per_image_ = image_embed @ text_embed.t()
    positive_logits_mean = logits_per_image_.diag().sum() / local_batch_size
    negative_logits_mean = (logits_per_image_.sum() - positive_logits_mean * local_batch_size) / (
            local_batch_size * local_batch_size - local_batch_size)
    return positive_logits_mean, negative_logits_mean


class RQVAE_EMBED_CLIP(nn.Module):
    def __init__(self,
                 hps,
                 ddconfig,
                 checkpointing,
                 ):
        super().__init__()

        self.rq_model = RQVAE_EMBED(**hps, ddconfig=ddconfig, checkpointing=True)

        self.logit_scale_self = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_cl = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # loss
        self.cl_head = CLIPLoss()

    def forward(self, *args, **kwargs):
        # 检查是否需要返回CLIP损失，默认不返回
        if not kwargs.get('return_clip_loss', False):
            # 确保只有一个位置参数被传递
            assert len(args) == 1
            # 调用RQ-VAE的前向传播方法
            return self.forward_rqvae(*args, **kwargs)
        # 如果需要返回CLIP损失，则确保有两个位置参数被传递
        assert len(args) == 2
        # 删除'return_clip_loss'关键字参数
        del kwargs['return_clip_loss']
        # 调用CLIP的前向传播方法
        return self.forward_clip(*args, **kwargs)

    def forward_rqvae(self, feat):
        out, loss_dict, selected_index, feature_norm, quant_norm, _, _, z_e = self.rq_model(feat, detail=True)
        loss = self.compute_loss(out, loss_dict['commitment_loss'], '', xs=feat, valid=False)
        loss_dict.update(loss)

        loss_unique = torch.tensor(0.).to(feat.device)
        loss_dict['loss_unique'] = loss_unique

        return loss['loss_total'] + loss_unique, out, selected_index, loss_dict, feature_norm, quant_norm

    def add_random_perturbation(self, original, sigma=0.001):
        """对向量数组进行正态分布的随机扰动"""
        noise = torch.randn(original.size(), device=original.device) * sigma
        return original + noise

    def compute_cosine_and_l2_mean(self, vec1, vec2):
        # 计算夹角
        cosine_similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1)
        cosine_similarity_mean = torch.mean(cosine_similarity)

        # 计算 L2 距离
        l2_distances = torch.norm(vec1 - vec2, dim=-1)
        l2_mean = torch.mean(l2_distances)

        return cosine_similarity_mean, l2_mean

    def forward_clip(self, fea1, fea2, **kwargs):
        fea1_vq, loss_dict1, selected_index1, feature_norm1, quant_norm1, z_q1, all_distances1, z_e1 = self.rq_model(
            fea1, detail=True, **kwargs)
        fea2_vq, loss_dict2, selected_index2, feature_norm2, quant_norm2, z_q2, all_distances2, z_e2 = self.rq_model(
            fea2, detail=True, **kwargs)

        # log重建相似度和loss
        feas = torch.cat([fea1, fea2], dim=0)
        recons = torch.cat([fea1_vq, fea2_vq], dim=0)

        # 对比学习
        features = {'image_embed': fea1_vq,
                    'text_embed': fea2_vq,
                    'image_embed_ori': fea1,
                    'text_embed_ori': fea2,
                    'logit_scale_self': self.logit_scale_self.exp(),
                    'logit_scale_cl': self.logit_scale_cl.exp(),
                    'logit_scale': self.logit_scale.exp(),
                    'image_embed2': z_e1,
                    'text_embed2': z_e2,
                    }
        ret = self.cl_head(features)

        # log
        ret['fea1_vq'] = fea1_vq

        ret['loss_dict1'] = loss_dict1
        ret['loss_dict2'] = loss_dict2
        ret['commitment_loss'] = (loss_dict1['commitment_loss'] + loss_dict2['commitment_loss']) / 2

        # log重建相似度和loss
        recon_loss = self.compute_loss(recons, ret['commitment_loss'], '', xs=feas, valid=False)['recon_loss']

        # 两个feat的相似度
        pair_code_loss = F.mse_loss(z_e1, z_e2, reduction='mean')

        ret.update({
            'recon_loss': recon_loss,
            'pair_code_loss': pair_code_loss,
        })

        return ret

    def compute_loss(self, out, quant_loss, code, xs=None, valid=False):
        return self.rq_model.compute_loss(out, quant_loss, code, xs=xs)

    @torch.no_grad()
    def get_decode_feature(self, xs):
        code = self.rq_model.get_codes(xs)
        out = self.rq_model.decode_code(code)
        return out

    def decode(self, z_q):
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q = z_q.contiguous()
        # z_q = self.post_quant_conv(z_q)
        out = self.decoder(z_q)
        out = F.normalize(out, p=2, dim=1)
        return out

    @torch.no_grad()
    def get_codes(self, xs):
        z_e = self.rq_model.encode(xs)
        _, _, code, _, _, _, _, _, all_distances = self.rq_model.quantizer(z_e)
        return code, all_distances

    @torch.no_grad()
    def get_sorted_index(self, xs):
        z_e = self.rq_model.encode(xs)
        _, _, code, _, _, _, _, _, all_distances = self.rq_model.quantizer(z_e)
        sorted_indices = torch.argsort(all_distances[-1], dim=-1) # level 3
        return code, sorted_indices

    @torch.no_grad()
    def get_soft_codes(self, xs, temp=1.0, stochastic=False):
        assert hasattr(self.quantizer, 'get_soft_codes')

        z_e = self.encode(xs)
        soft_code, code = self.quantizer.get_soft_codes(z_e, temp=temp, stochastic=stochastic)
        return soft_code, code

    @torch.no_grad()
    def decode_code(self, code):
        z_q = self.quantizer.embed_code(code)
        decoded = self.decode(z_q)
        return decoded

    def get_recon_imgs(self, xs_real, xs_recon):
        xs_real = xs_real * 0.5 + 0.5
        xs_recon = xs_recon * 0.5 + 0.5
        xs_recon = torch.clamp(xs_recon, 0, 1)

        return xs_real, xs_recon

    def cosine_loss(self, x1, x2):
        cos_sim = F.cosine_similarity(x1, x2, dim=1)
        loss = 1 - cos_sim
        return loss.mean()

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def get_code_emb_with_depth(self, code):
        return self.quantizer.embed_code_with_depth(code)

    @torch.no_grad()
    def decode_partial_code(self, code, code_idx, decode_type='select'):
        r"""
        Use partial codebooks and decode the codebook features.
        If decode_type == 'select', the (code_idx)-th codebook features are decoded.
        If decode_type == 'add', the [0,1,...,code_idx]-th codebook features are added and decoded.
        """
        z_q = self.quantizer.embed_partial_code(code, code_idx, decode_type)
        decoded = self.decode(z_q)
        return decoded

    @torch.no_grad()
    def forward_partial_code(self, xs, code_idx, decode_type='select'):
        r"""
        Reconstuct an input using partial codebooks.
        """
        code = self.get_codes(xs)
        out = self.decode_partial_code(code, code_idx, decode_type)
        return out


class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        image_embed_ori = outputs['image_embed_ori']
        text_embed_ori = outputs['text_embed_ori']
        logit_scale = outputs['logit_scale']
        logit_scale_self = outputs['logit_scale_self']
        logit_scale_cl = outputs['logit_scale_cl']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * dist_utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather with gpu
        image_embed_all, text_embed_all = dist_utils.all_gather_batch_with_grad([image_embed, text_embed])
        image_embed_all_ori, text_embed_all_ori = dist_utils.all_gather_batch_with_grad(
            [image_embed_ori, text_embed_ori])

        # ------------------------------------------------------------------------------

        # loss self
        logits_per_image = logit_scale_self * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale_self * text_embed @ image_embed_all.t()

        # loss ori
        logits_per_image2 = logit_scale * image_embed @ text_embed_all_ori.t()
        logits_per_text2 = logit_scale * text_embed @ image_embed_all_ori.t()

        # loss cl
        logits_per_image3 = logit_scale_cl * image_embed @ image_embed_all_ori.t()
        logits_per_text3 = logit_scale_cl * text_embed @ text_embed_all_ori.t()

        loss_self = (F.cross_entropy(logits_per_image, self.labels) + F.cross_entropy(logits_per_text, self.labels)) / 2
        loss_ori = (F.cross_entropy(logits_per_image2, self.labels) + F.cross_entropy(logits_per_text2,
                                                                                      self.labels)) / 2
        loss_cl = (F.cross_entropy(logits_per_image3, self.labels) + F.cross_entropy(logits_per_text3, self.labels)) / 2

        loss = (loss_ori + loss_self + loss_cl) / 3

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pred = torch.argmax(logits_per_text, dim=-1)
            correct2 = pred.eq(self.labels).sum()
            pred = torch.argmax(logits_per_image2, dim=-1)
            correct3 = pred.eq(self.labels).sum()
            pred = torch.argmax(logits_per_text2, dim=-1)
            correct4 = pred.eq(self.labels).sum()
            acc = 100 * (correct + correct2 + correct3 + correct4) / local_batch_size / 4

        return {'loss': loss,
                'clip_loss': loss,
                'clip_acc': acc,
                'loss_ori': loss_ori,
                'loss_self': loss_self,
                'loss_cl': loss_cl,
                }
