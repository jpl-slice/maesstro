# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import re
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.input_channels = in_chans
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *C)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.input_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.input_channels))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.input_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.input_channels, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def apply_patch_mask(self, x, patch_mask):
        """
        Given a patch mask (e.g., from real clouds), apply it to the sequence of patches
        instead of randomly selecting patches to mask.
        x: [N, L, D], sequence
        patch_mask: [N, L], binary mask
        """
        N, L, D = x.shape
        ids_keep = torch.where(patch_mask == 0)[1]
        ids_mask = torch.where(patch_mask == 1)[1]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # ids_restore is the argsort of the concatneated ids_keep and ids_mask
        ids_restore = (
            torch.argsort(torch.cat([ids_keep, ids_mask]), dim=0)
            .unsqueeze(0)
            .repeat(N, 1)
            .to(x.device)
        )
        return x_masked, patch_mask, ids_restore

    def forward_encoder(self, x, mask_ratio, patch_mask=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if patch_mask is None:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.apply_patch_mask(x, patch_mask)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # replicate cls_token across each image in batch
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence (mask token shape = (1, 1, decoder_embed_dim))
        # repeat mask token for each masked patch across each sample in batch
        num_masked = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_random_mask_ratio(self, images):
        # embed patches
        patchified_images = self.patch_embed(images)
        patchified_images = (
            patchified_images + self.pos_embed[:, 1:, :]
        )  # take second pos embedding onwards because x doesn't have cls_token yet

        # for each image in x, randomly select patches to pass through encoder block
        preds = torch.zeros(
            images.shape[0],
            patchified_images.shape[1],
            self.decoder_pred.out_features,
            device=images.device,
        )
        masks = torch.zeros(
            images.shape[0], patchified_images.shape[1], device=images.device
        )
        for i, img_patch_embedding in enumerate(patchified_images):
            # randomly select patches to mask
            mask_ratio = torch.rand(1, device=img_patch_embedding.device) * 0.8 + 0.1
            x, mask, ids_restore = self.random_masking(
                img_patch_embedding.unsqueeze(0), mask_ratio
            )  # mask.shape should be (1, num_patches (L))
            masks[i] = mask

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            # run encoder blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            # run decoder
            x = self.decoder_embed(x)
            num_masked = ids_restore.shape[1] + 1 - x.shape[1]
            mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
            )  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            x = x + self.decoder_pos_embed

            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            x = self.decoder_pred(x)
            x = x[:, 1:, :]  # remove cls token
            preds[i] = x

        loss, r2 = self.forward_loss(images, preds, masks)
        return masks, preds, loss, r2

    def forward_random_mask_ratio_vectorized(self, images):
        """
        `forward_random_mask_ratio` runs MAE's forward pass on a batch of images, where each image in the batch has a random masking ratio. 
        This operation is really slow because we can't vectorize the forward pass across all images in the batch, due to the fact that each image ends up with a different number of unmasked tokens that we feed into the MAE encoder blocks.
        Here, we attempt to make this operation faster by enabling some vectorization. Even if we can't vectorize the entire batch due to different shapes,
        we split up the mini-batch into mini-mini-batches containing samples of the same masking ratios and run the forward pass on that.
        
        Here's how we do it:
        1. Generate a dictionary of masking ratios to indices in the mini-batch. Each key in the dictionary is a masking ratio, and each value is a list of indices in the mini-batch that have that masking ratio.
        Each list should have roughly the same number of indices. If the batch size isn't divisible by the number of masking ratios, then we uniformly sample the remaining indices to add to the dictionary.
        The masking ratios I want to use (i.e., dictionary keys) are: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        2. For each masking ratio, run the forward pass on the mini-mini-batch of images that have that masking ratio.
        3. Re-assemble the predictions and masks into the original order of the mini-batch.
        4. Run the self.forward_loss() function on the predictions and masks.
        """
        # embed patches
        patchified_images = self.patch_embed(images)
        patchified_images = (
            patchified_images + self.pos_embed[:, 1:, :]
        )  # take second pos embedding onwards because x doesn't have cls_token yet

        # for each image in x, randomly select patches to pass through encoder block
        preds = torch.zeros(
            images.shape[0],
            patchified_images.shape[1],
            self.decoder_pred.out_features,
            device=images.device,
        )
        masks = torch.zeros(
            images.shape[0], patchified_images.shape[1], device=images.device
        )

        # generate dictionary of masking ratios to indices in the mini-batch
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        shuffled_indices = torch.randperm(images.shape[0])
        ratio_to_indices = {}
        num_images_per_masking_ratio = images.shape[0] // len(ratios)
        # assign each index in the mini-batch to a masking ratio
        for i, ratio in enumerate(ratios):
            ratio_to_indices[ratio] = shuffled_indices[
                i
                * num_images_per_masking_ratio : (i + 1)
                * num_images_per_masking_ratio
            ]
        # deal with remainder
        remainder = images.shape[0] % len(ratios)
        if remainder > 0:
            for i in range(remainder):
                random_ratio = ratios[torch.randint(0, len(ratios), (1,))]
                ratio_to_indices[random_ratio] = torch.cat(
                    [
                        ratio_to_indices[random_ratio],
                        shuffled_indices[-(i + 1)].unsqueeze(0),
                    ]
                )

        # for each masking ratio, run the forward pass on the mini-mini-batch of images that have that masking ratio
        for ratio, indices in ratio_to_indices.items():
            mini_latent, mini_mask, mini_ids_restore = self.forward_encoder(
                images[indices], ratio
            )
            pred = self.forward_decoder(mini_latent, mini_ids_restore)
            preds[indices] = pred.type(preds.dtype)
            masks[indices] = mini_mask

        # Calculate loss
        loss, r2 = self.forward_loss(images, preds, masks)
        return masks, preds, loss, r2

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*C]  # L is the number of patches
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # shape: [N, L, p*p*C]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        squared_error = (pred - target) ** 2
        loss = squared_error.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # also calculate R2 score
        ss_res = squared_error.sum(dim=-1)  # [N, L], sum of squared error per patch
        ss_tot = ((target - target.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)  # [N, L]
        # for each image, sum across patches:
        ss_tot = (ss_tot * mask).sum(dim=-1, keepdim=True) + 1.0e-6  # [N, 1]
        r2 = 1 - (ss_res / ss_tot)
        r2 = (r2 * mask).sum() / mask.sum()  # mean R2 on removed patches
        return loss, r2

    def forward(self, imgs, mask_ratio=0.75, patch_mask=None):
        mask, pred, loss, r2 = self.forward_random_mask_ratio_vectorized(imgs)
        loss, r2 = self.forward_loss(imgs, pred, mask)
        return loss, r2, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# MaskedAutoencoderViT kwargs:
#  img_size=224, patch_size=16, in_chans=3,
#  embed_dim=1024, depth=24, num_heads=16,
#  decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#  mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False


def mae_vit_base_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch12_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=12,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch4_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_tiny_patch4_dec512d8b(**kwargs):
    # https://github.com/huggingface/pytorch-image-models/blob/cd3ee78387e2c6d7b68882c79c4b988635fef294/timm/models/vision_transformer.py#L1321
    model = MaskedAutoencoderViT(
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_builder(model_name, **kwargs):
    """Builds a MaskedAutoencoderViT model by name.
    For example, "mae_vit_base_patch16" means
    patch_size = 16
    embed_dim = 768
    depth = 12
    num_heads = 12
    """
    vit_size_map = {  # maps size to (embed_dim, depth, num_heads)
        "huge": (1280, 32, 16),
        "large": (1024, 24, 16),
        "base": (768, 12, 12),
        "small": (384, 12, 6),
        "tiny": (192, 12, 3),
    }

    # get size from model_name and set kwargs
    size = model_name.split("_")[2]
    kwargs["embed_dim"], kwargs["depth"], kwargs["num_heads"] = vit_size_map[size]

    # get patch_size from model_name and set kwargs
    kwargs["patch_size"] = int(model_name.split("_")[3][5:])

    # get decoder_embed_dim and decoder_depth from model_name and set kwargs
    decoder_substring = model_name.split("_dec")[1]
    if len(decoder_substring) > 0:  # example decoder substring: 512d8b16h
        kwargs.update(extract_numbers(decoder_substring))
    else:
        kwargs["decoder_embed_dim"] = 512
        kwargs["decoder_depth"] = 8
        kwargs["decoder_num_heads"] = 16

    # get decoder_num_heads from model_name and set kwargs
    print(f"Initializing MaskedAutoencoderViT with kwargs: {kwargs}")
    model = MaskedAutoencoderViT(**kwargs)
    return model


def extract_numbers(decoder_substring):
    result = {}
    matches = re.findall(r"\d+", decoder_substring)
    if len(matches) >= 2:
        result["decoder_embed_dim"] = int(matches[0])
        result["decoder_depth"] = int(matches[1])
    if len(matches) == 3:
        result["decoder_num_heads"] = int(matches[2])
    else:
        result["decoder_num_heads"] = 16
    return result


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch12 = mae_vit_base_patch12_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch8 = mae_vit_large_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch4 = mae_vit_base_patch4_dec512d8b  # decoder: 512 dim, 8 blocks
