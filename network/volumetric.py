# Reference : https://github.com/karfly/learnable-triangulation-pytorch

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.backbone import get_backbone
from network.v2v import V2VModel

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def volumetric_cube(batch_size, bound_box_size, coord_cube_size, pelvis, training):
    '''make volumetric cube'''
    coord_cubes = torch.zeros(batch_size, coord_cube_size, coord_cube_size, coord_cube_size, 3, device=device)
    # calculate volumetric cube per batch
    for b in range(batch_size):
        # pelvis position from ground truth (np.array)
        pelvis_position = pelvis[b]
        pelvis_position = (pelvis_position[11, :3] + pelvis_position[12, :3]) / 2

        # 3D bound box position
        cube_sides    = np.array([bound_box_size, bound_box_size, bound_box_size]) # (size: L x L x L)
        cube_position = pelvis_position - cube_sides / 2

        # coordinate cube
        x, y, z = torch.meshgrid(torch.arange(coord_cube_size).to(device),
                                 torch.arange(coord_cube_size).to(device),
                                 torch.arange(coord_cube_size).to(device))
        grid_mesh = torch.stack([x, y, z], dim=1).type(torch.float).reshape((-1, 3))

        # discretize the bound box by volumetric cube filling with the global coordinates
        coord_cube = torch.zeros_like(grid_mesh)
        coord_cube[:, 0] = cube_position[0] + (cube_sides[0] / (coord_cube_size - 1)) * grid_mesh[:, 0] # X-axis
        coord_cube[:, 1] = cube_position[1] + (cube_sides[1] / (coord_cube_size - 1)) * grid_mesh[:, 1] # Y-axis
        coord_cube[:, 2] = cube_position[2] + (cube_sides[2] / (coord_cube_size - 1)) * grid_mesh[:, 2] # Z-axis
        coord_cube = coord_cube.reshape((coord_cube_size, coord_cube_size, coord_cube_size, 3))
            
        # Y-axis perpendicular to the ground & X-axis random orientation(only training)
        if training:
            theta = np.random.uniform(0.0, 2*np.pi) # radians,  0 <= theta <= 2pi
        else:
            theta = 0.0
        axis = np.array([0, 1, 0]) # Y-axis
        center = torch.from_numpy(pelvis_position).type(torch.float).to(device)
        coord_cube = coord_cube - center
        coord_cube = rotate_cube(coord_cube, theta, axis)
        coord_cube = coord_cube + center

        coord_cubes[b] = coord_cube

    return coord_cubes


def rotate_cube(cube, theta, axis):
    shape = cube.shape
    # rotation  matrix
    unit_axis = axis / np.sqrt(np.dot(axis, axis))
    x, y, z = unit_axis
    cos, sin = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos+(x**2)*(1-cos), x*y*(1-cos)-z*sin,  x*z*(1-cos)+y*sin],
                           [y*x*(1-cos)+z*sin,  cos+(y**2)*(1-cos), y*z*(1-cos)-x*sin],
                           [z*x*(1-cos)-y*sin,  z*y*(1-cos)+x*sin,  cos+(z**2)*(1-cos)]])
    rot_matrix = torch.from_numpy(rot_matrix).type(torch.float).to(device)

    # rotate
    cube = cube.view(-1, 3).t()
    cube = torch.einsum("ij, jk -> ik", rot_matrix, cube).t()
    cube = cube.view(*shape)

    return cube


def unprojection(heatmaps, projection_matrices, coord_cubes, vol_confidences, agg_type):
    '''
    inputs : 
        heatmaps - intermedate heatmaps (size: B, C, K=32, H_h, W_h)
        projection_matrices             (size: B, C, 3, 4)
        coord_cubes                     (size: B, 64, 64, 64, 3)
        vol_confidences                 (size: B, C, K=32)
        type = ('sum', 'conf', 'softmax')
    output :
        volumes                         (size: B, K, 64, 64, 64)
    '''
    B, C, K, H_h, W_h = heatmaps.shape
    volumes = torch.zeros(B, K, *coord_cubes.shape[1:4], device=device)

    # calculate per batch, per cameras
    for b in range(B):
        coord_grid = coord_cubes[b].reshape((-1, 3))
        volumes_batch = torch.zeros(C, K, *coord_cubes.shape[1:4], device=device)
        for c in range(C):
            heatmap = heatmaps[b, c].unsqueeze(0) # (size: 1, 32, H_h, W_h)
            # project the 3D coordinates to the plane
            coord_grid_homo = torch.cat([coord_grid, torch.ones((coord_grid.shape[0], 1), dtype=coord_grid.dtype, device=coord_grid.device)], dim=1)
            coord_grid_proj_homo = torch.matmul(coord_grid_homo, projection_matrices[b, c].t()) # (size: 64*64*64, 3)
            invalid = coord_grid_proj_homo[:, 2] <= 0.0
            coord_grid_proj_homo[coord_grid_proj_homo[:, 2] == 0.0, 2] = 1.0
            coord_grid_proj = coord_grid_proj_homo[:, :-1] / coord_grid_proj_homo[:, -1:] # (size: 64*64*64, 2)

            # transform to [-1.0, 1.0] range
            coord_grid_proj_transformed = torch.zeros_like(coord_grid_proj)
            coord_grid_proj_transformed[:, 0] = 2 * (coord_grid_proj[:, 0] / H_h - 0.5)
            coord_grid_proj_transformed[:, 1] = 2 * (coord_grid_proj[:, 1] / W_h - 0.5)
            coord_grid_proj = coord_grid_proj_transformed

            # bilinear sampling from the heatmaps of the corresponding camera view using coord_grid_proj
            try:
                volume_view = F.grid_sample(heatmap, coord_grid_proj.unsqueeze(1).unsqueeze(0), align_corners=True) # (size: 1, 32, 64*64*64, 1)
            except TypeError: # old version
                volume_view = F.grid_sample(heatmap, coord_grid_proj.unsqueeze(1).unsqueeze(0))
            #volume_view = F.grid_sample(heatmap, coord_grid_proj.unsqueeze(1).unsqueeze(0), align_corners=True) # (size: 1, 32, 64*64*64, 1)
            volume_view = volume_view.view(K, -1)
            volume_view[:, invalid] = 0.0
            volume_view = volume_view.view(K, *coord_cubes.shape[1:4]) #(size: 32, 64, 64, 64)
            volumes_batch[c] = volume_view

        # aggregate the volumetric maps from all views
        if agg_type == 'sum':
            volumes[b] = volumes_batch.sum(0)
        elif agg_type == 'conf':
            volumes[b] = (volumes_batch * vol_confidences[b].view(C, K, 1, 1, 1)).sum(0)
        elif agg_type == 'softmax':
            volumes_batch_weight = nn.Softmax(0)(volumes_batch)
            volumes[b] = (volumes_batch_weight * volumes_batch).sum(0)

    return volumes



class VolumetricTriangulation(nn.Module):
    '''
    VolumetricTriangulation : 2D backbone -> Unprojection -> V2V -> Softmax operation -> 3D pose keypoists

    B = number of batches
    C = number of cameras = 4
    J = number of joints  = 17
    H, W = height, width of images

    input : 
        images             (size: B, C, 3, H, W)
        prjection_matrices (size: B, C, 3, 4)
        batch - type: dict
    output :
        keypoints_3d       (size: B, J, 3)
    '''
    def __init__(self, args):
        super(VolumetricTriangulation, self).__init__()

        self.backbone = get_backbone(args, alg_confidences=False, vol_confidences=True)
        self.singleConv = nn.Sequential(nn.Conv2d(256, 32, 1))
        self.bound_box_size = 2500.0
        self.coord_cube_size = 64
        self.agg_type = args.agg_type
        self.volumeConv = V2VModel(32, 17)


    def forward(self, images, projection_matrices, pelvis):
        # reshape images
        B, C, _, H, W = images.shape
        images = images.view(-1, *images.shape[2:])

        # 2D backbone
        _, intermediate_heatmaps, _, vol_confidences = self.backbone(images)
        intermediate_heatmaps = self.singleConv(intermediate_heatmaps) # (size: BxC, 32, H/4, W/4)

        # reshape & normalize
        intermediate_heatmaps = intermediate_heatmaps.view(B, C, *intermediate_heatmaps.shape[1:])
        vol_confidences = vol_confidences.view(B, C, *vol_confidences.shape[1:])
        if self.agg_type == 'conf':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # volumetric cube (size: B, 64, 64, 64, 3)
        coord_cubes = volumetric_cube(B, self.bound_box_size, self.coord_cube_size, pelvis, self.training)

        # aggregated volumes (size : B, 32, 64, 64, 64)
        volumes = unprojection(intermediate_heatmaps, projection_matrices, coord_cubes, vol_confidences, self.agg_type)
        
        # volumetric convolutional neural network (V2V) (size : B, 17, 64, 64, 64)
        volumes = self.volumeConv(volumes)
        
        # softmax operation
        volumes_w = nn.Softmax(2)(volumes.reshape((B, C, -1))).reshape(*volumes.shape)
        keypoints_3d = torch.einsum("bjxyz, bxyzc -> bjc", volumes_w, coord_cubes)

        return keypoints_3d