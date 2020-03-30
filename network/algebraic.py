# Reference : https://github.com/karfly/learnable-triangulation-pytorch
# Reference : (BOOK) "Multiple view geometry in computer vision" Richard Hartley and Andrew Zisserman (P. 312, P.592)

import numpy as np

import torch
import torch.nn as nn

from network.backbone import get_backbone

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def softmax_operation(heatmaps, inverse_temperature=100):
    '''
    compute the softmax across the spatial axes
    &
    calculate the 2D positions of the joints as the center of mass of the heatmaps(softargmax)
    B' = Batch_size x Number of Camera = BxC
    input  : 2D joint heatmaps  (size: B', J, H, W)
    output : 2D joint keypoints (size: B', J, 2)
             2D joint heatmaps  (size: B', J, H, W)
    '''
    # softmax
    heatmaps = heatmaps*inverse_temperature
    b, j, h, w = heatmaps.shape
    heatmaps = nn.Softmax(2)(heatmaps.reshape((b, j, -1))).reshape((b, j, h, w))

    # soft-argmax
    x, y = torch.meshgrid(torch.arange(w), torch.arange(h))
    grid_2d = torch.stack([x, y], dim=-1).type(torch.float).to(device)
    keypoints_2d = torch.einsum("bjhw, whc -> bjc", heatmaps, grid_2d)

    return keypoints_2d, heatmaps


def DLT(projection_matrices, keypoints_2d, confidences):
    '''
    DLT(Direct Linear Transformation) - Homogeneous method
    Human3.6M : number of cameras = 4
    input :
        projection_matrices (size: B, C, 3, 4) - sequences of camera projection matrices(3x4)
        keypoints_2d        (size: B, C, J, 2)
        confidences         (size: B, C, J)
    output : 
        keypoints_3d        (size: B, J, 3)
    '''
    B, C, J = keypoints_2d.shape[:3]
    keypoints_3d = torch.zeros((B, J, 3))
    for b in range(B):
        for j in range(J):

            keypoints = keypoints_2d[b, :, j, :] # size: C, 2
            confidence = confidences[b, :, j]    # size: C
            p_matrices = projection_matrices[b]  # size: C, 3, 4

            # A : matrix composed of the components from the full projection matrices & keypoints_2d
            A = p_matrices[:, 2:3, :].expand(C, 2, 4) * keypoints.view(C, 2, 1)
            A = A - p_matrices[:, :2, :]
            A = A * confidence.view(-1, 1, 1)

            # solving equation ((w*A)y=0), w*A=UDV^T, y is the last column of V
            u, s, vt = torch.svd(A.view(-1, 4))
            keypoints_homo = vt[:, 3] # size: 4, 

            # homogeneous y to non-homogeneous y
            keypoints_nonhomo = keypoints_homo[:-1] / keypoints_homo[-1] # size : 3
            keypoints_3d[b, j] = keypoints_nonhomo
    
    return keypoints_3d


class AlgebraicTriangulation(nn.Module):
    '''
    AlgebraicTriangulation : 2D backbone -> Softmax operation -> DLT -> 3D pose keypoists
    
    B = number of batches
    C = number of cameras
    J = number of joints
    H, W = height, width of images

    input  : torch tensor
        images             (size: B, C, 3, H, W)
        prjection_matrices (size: B, C, 3, 4)
    output : torch tensor
        keypoints_3d       (size: B, J, 3)
    '''
    def __init__(self, args):
        super(AlgebraicTriangulation, self).__init__()

        self.backbone = get_backbone(args, alg_confidences=True, vol_confidences=False)


    def forward(self, images, projection_matrices):
        # reshape images
        B, C, _, H, W = images.shape
        images = images.view(-1, *images.shape[2:])

        # 2D backbone & softmax operation
        interpretable_heatmaps, _, alg_confidences, _ = self.backbone(images)
        keypoints_2d, _ = softmax_operation(interpretable_heatmaps)

        # reshape 
        keypoints_2d = keypoints_2d.view(B, C, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(B, C, *alg_confidences.shape[1:])

        # upscale
        keypoints_2d[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (H / interpretable_heatmaps.shape[2])
        keypoints_2d[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (W / interpretable_heatmaps.shape[3])

        # algebraic trangulation (DLT)
        keypoints_3d = DLT(projection_matrices, keypoints_2d, alg_confidences)

        return keypoints_3d


