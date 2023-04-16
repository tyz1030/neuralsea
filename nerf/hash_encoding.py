import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        print("ALERT: some points are outside bounding box. Clipping them!")
        pdb.set_trace()
        # print(xyz.shape)
        # print((xyz > box_max).nonzero()[0])
        # i = (xyz > box_max).nonzero()[0]
        # print((xyz < box_min).nonzero())
        # print(xyz[i[0],i[1],i[2],i[3],:])
        import sys
        sys.exit()
        xyz = torch.clamp(xyz, min=box_min, max=box_max)
    # print(torch.max(xyz[0, :, -1, 0, 0]))
    # print(xyz.shape)
    grid_size = (box_max-box_min)/resolution
    
    bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
    voxel_min_vertex = bottom_left_idx*grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0]).cuda()*grid_size

    # print(bottom_left_idx.unsqueeze(-2).shape)
    # print(BOX_OFFSETS.shape)
    voxel_indices = bottom_left_idx.unsqueeze(-2) + BOX_OFFSETS
    # print(voxel_indices.shape)
    # import sys
    # sys.exit()
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

class HashEmbedder(nn.Module):
    # def __init__(self, bounding_box = (torch.tensor([-1.2, -1.2, -1.2]).cuda(), torch.tensor([1.2, 1.2, 1.2]).cuda()), n_levels=16, n_features_per_level=2,\
    def __init__(self, bounding_box = (torch.tensor([-1.4, -1.4, -1.4]).cuda(), torch.tensor([1.4, 1.4, 1.4]).cuda()), n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # print(voxel_embedds.shape)
        # print(weights[:,0][:,None].shape)
        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[...,0, :]*(1-weights[...,0][...,None]) + voxel_embedds[...,4, :]*weights[...,0][...,None]
        c01 = voxel_embedds[...,1, :]*(1-weights[...,0][...,None]) + voxel_embedds[...,5, :]*weights[...,0][...,None]
        c10 = voxel_embedds[...,2, :]*(1-weights[...,0][...,None]) + voxel_embedds[...,6, :]*weights[...,0][...,None]
        c11 = voxel_embedds[...,3, :]*(1-weights[...,0][...,None]) + voxel_embedds[...,7, :]*weights[...,0][...,None]

        # step 2
        c0 = c00*(1-weights[...,1][...,None]) + c10*weights[...,1][...,None]
        c1 = c01*(1-weights[...,1][...,None]) + c11*weights[...,1][...,None]

        # step 3
        c = c0*(1-weights[...,2][...,None]) + c1*weights[...,2][...,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            # print(voxel_min_vertex.shape)
            # print(voxel_max_vertex.shape)
            # print(hashed_voxel_indices.shape)
            # print(voxel_embedds.shape)
            # import sys
            # sys.exit()

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

