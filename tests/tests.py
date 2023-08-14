import pytest

import torch
from einops.layers.torch import Rearrange, Reduce

from nerfstudio.cameras.rays import RaySamples, Frustums

from reni.utils.utils import find_nerfstudio_project_root, rot_z, rot, rot_y
from reni.discriminators.discriminators import VNTransformerDiscriminatorConfig
from reni.field_components.vn_layers import VNInvariant, VNLayerNorm


# torch.set_default_dtype(torch.float64)

# test invariant layers
def test_vn_discriminator_invariance():
    model = VNTransformerDiscriminatorConfig(invariance='SO3',
                                             return_intermediate_components=True,
                                             fusion_strategy="late").setup()
    model = model
  
    batch_size = 2
    num_rays = 32
    coors = torch.randn(batch_size, num_rays, 3)
    colours = torch.randn(batch_size, num_rays, 3)

    ray_samples = RaySamples(frustums=Frustums(origins=torch.zeros_like(coors),
                                               directions=coors,
                                               starts=torch.zeros_like(coors[:, :, 0:1]),
                                                ends=torch.ones_like(coors[:, :, 0:1]),
                                                pixel_area=torch.ones_like(coors[:, :, 0:1])),
                              camera_indices=torch.zeros_like(coors[:, :, 0:1]))
    
    out, (x1, x2, x3, x4) = model(ray_samples, colours)
    # R = rot(*torch.randn(3))
    R = rot_z(*torch.randn(1))
    # R = rot_y(*torch.randn(1))
    ray_samples.frustums.directions = ray_samples.frustums.directions @ R
    out_rot, (x1_rot, x2_rot, x3_rot, x4_rot) = model(ray_samples, colours)

    # rotate the directional component of x1
    x1[:, :, :3] = x1[:, :, :3] @ R
    print("Testing equivaraince of x1")
    print(torch.allclose(x1, x1_rot, atol = 1e-6))

    # rotate the directional component of x2
    x2[:, :, :, :3] = x2[:, :, :, :3] @ R
    print("Testing equivaraince of x2")
    print(torch.allclose(x2, x2_rot, atol = 1e-6))

    # rotate the directional component of x3
    x3_r_test = x3[:, :, :, :3] @ R
    print("Testing equivaraince of x3")
    print(torch.allclose(x3_r_test, x3_rot, atol = 1e-5))
    
    print(x3.shape)
    print(x3[0, :5, 0, :3])
    print(x3_rot[0, :5, 0, :3])

    hidden_dim = x3.shape[2]
    vn_invariant = VNInvariant(hidden_dim, dim_coor=6)
    vn_layer_norm = VNLayerNorm(hidden_dim)
    x3 = vn_layer_norm(x3)
    x3_rot = vn_layer_norm(x3_rot)
    # testing if we can make SO(2) invariance
    # take out the z component of x3
    x3_z = x3[:, :, :, 2:3] # [B, N, F, 1]
    x4 = vn_invariant(x3) # [B, N, 6]
    # reduce x3_z from [B, N, F, 1] to [B, N, 1]
    x4_z = Reduce('b n f c -> b n c', 'mean')(x3_z)
    # swap z component of x4 with x4_z
    x4[:, :, 2:3] = x4_z

    # same again for rotated x3
    x3_rot_z = x3_rot[:, :, :, 2:3] # [B, N, F, 1]
    x4_rot = vn_invariant(x3_rot) # [B, N, 6]
    # reduce x3_z from [B, N, F, 1] to [B, N, 1]
    x4_rot_z = Reduce('b n f c -> b n c', 'mean')(x3_rot_z)
    # swap z component of x4 with x4_z
    x4_rot[:, :, 2:3] = x4_rot_z

    # print first 5 z components of x4
    print(x4[0, :5, :3])
    print(x4_rot[0, :5, :3])
    
    # output x4 is invariant to rotation
    print("Testing invariance of x4")
    print(torch.allclose(x4, x4_rot, atol = 1e-5))


    

    # # get the directional components of out_invar and out_invar_rot
    # out_invar = out_invar[:, :, :3]
    # out_invar_rot = out_invar_rot[:, :, :3]
    # print(torch.allclose(out_invar, out_invar_rot, atol = 1e-6))

    # assert torch.allclose(out_invar, out_invar_rot, atol = 1e-6)

# calls as main
if __name__ == '__main__':
    test_vn_discriminator_invariance()


# test equivariance

# @pytest.mark.parametrize('l2_dist_attn', [True, False])
# def test_equivariance(l2_dist_attn):

#     model = VNTransformer(
#         dim = 64,
#         depth = 2,
#         dim_head = 64,
#         heads = 8,
#         l2_dist_attn = l2_dist_attn
#     )

#     coors = torch.randn(1, 32, 3)
#     mask  = torch.ones(1, 32).bool()

#     R   = rot(*torch.randn(3))
#     out1 = model(coors @ R, mask = mask)
#     out2 = model(coors, mask = mask) @ R

#     assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# test vn perceiver attention equivariance

# @pytest.mark.parametrize('l2_dist_attn', [True, False])
# def test_perceiver_vn_attention_equivariance(l2_dist_attn):

#     model = VNAttention(
#         dim = 64,
#         dim_head = 64,
#         heads = 8,
#         num_latents = 2,
#         l2_dist_attn = l2_dist_attn
#     )

#     coors = torch.randn(1, 32, 64, 3)
#     mask  = torch.ones(1, 32).bool()

#     R   = rot(*torch.randn(3))
#     out1 = model(coors @ R, mask = mask)
#     out2 = model(coors, mask = mask) @ R

#     assert out1.shape[1] == 2
#     assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# # test SO(3) early fusion equivariance

# @pytest.mark.parametrize('l2_dist_attn', [True, False])
# def test_equivariance_with_early_fusion(l2_dist_attn):

#     model = VNTransformer(
#         dim = 64,
#         depth = 2,
#         dim_head = 64,
#         heads = 8,
#         dim_feat = 64,
#         l2_dist_attn = l2_dist_attn
#     )

#     feats = torch.randn(1, 32, 64)
#     coors = torch.randn(1, 32, 3)
#     mask  = torch.ones(1, 32).bool()

#     R   = rot(*torch.randn(3))
#     out1, _ = model(coors @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

#     out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
#     out2 = out2 @ R

#     assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# # test SE(3) early fusion equivariance

# @pytest.mark.parametrize('l2_dist_attn', [True, False])
# def test_se3_equivariance_with_early_fusion(l2_dist_attn):

#     model = VNTransformer(
#         dim = 64,
#         depth = 2,
#         dim_head = 64,
#         heads = 8,
#         dim_feat = 64,
#         translation_equivariance = True,
#         l2_dist_attn = l2_dist_attn
#     )

#     feats = torch.randn(1, 32, 64)
#     coors = torch.randn(1, 32, 3)
#     mask  = torch.ones(1, 32).bool()

#     T   = torch.randn(3)
#     R   = rot(*torch.randn(3))
#     out1, _ = model((coors + T) @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

#     out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
#     out2 = (out2 + T) @ R

#     assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'