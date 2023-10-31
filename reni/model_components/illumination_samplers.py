# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Samplers for spherical illumination fields.
"""

from abc import abstractmethod
from typing import Optional, Type
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.cameras.cameras import Cameras, CameraType


# Field related configs
@dataclass
class IlluminationSamplerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IlluminationSampler)
    """target class to instantiate"""


class IlluminationSampler(nn.Module):
    """Generate Samples

    Args:
        num_samples: number of samples to take
    """

    def __init__(
        self,
        config: IlluminationSamplerConfig,
    ) -> None:
        super().__init__()

    @abstractmethod
    def generate_direction_samples(self, num_directions: Optional[int] = None, apply_random_rotation=None) -> torch.Tensor:
        """Generate Direction Samples"""

    def forward(self, num_directions: Optional[int] = None, apply_random_rotation=None) -> torch.Tensor:
        """Returns directions for each position.

        Args:
            num_directions: number of directions to sample

        Returns:
            directions: [num_directions, 3]
        """

        return self.generate_direction_samples(num_directions, apply_random_rotation)


# Field related configs
@dataclass
class IcosahedronSamplerConfig(IlluminationSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: IcosahedronSampler)
    """target class to instantiate"""
    num_directions: int = 100
    """number of directions to sample"""
    apply_random_rotation: bool = False
    """apply random rotation to the icosphere"""
    remove_lower_hemisphere: bool = False
    """remove lower hemisphere"""


class IcosahedronSampler(IlluminationSampler):
    """For sampling directions from an icosahedron."""

    def __init__(
        self,
        config: IcosahedronSamplerConfig,
    ):
        super().__init__(config)
        self._num_directions = config.num_directions
        self.apply_random_rotation = config.apply_random_rotation
        self.remove_lower_hemisphere = config.remove_lower_hemisphere

        vertices, _ = self.icosphere(nr_verts=self._num_directions)
        self.directions = torch.from_numpy(vertices).float()  # [N, 3], # Z is up

    def icosphere(self, nu = 1, nr_verts = None):
        '''
        Returns a geodesic icosahedron with subdivision frequency nu. Frequency
        nu = 1 returns regular unit icosahedron, and nu>1 preformes subdivision.
        If nr_verts is given, nu will be adjusted such that icosphere contains
        at least nr_verts vertices. Returned faces are zero-indexed!
            
        Parameters
        ----------
        nu : subdivision frequency, integer (larger than 1 to make a change).
        nr_verts: desired number of mesh vertices, if given, nu may be increased.
            
        
        Returns
        -------
        subvertices : vertex list, numpy array of shape (20+10*(nu+1)*(nu-1)/2, 3)
        subfaces : face list, numpy array of shape (10*n**2, 3)
        
        '''
      
        # Unit icosahedron
        (vertices,faces) = self.icosahedron()

        # If nr_verts given, computing appropriate subdivision frequency nu.
        # We know nr_verts = 12+10*(nu+1)(nu-1)
        if not nr_verts is None:
            nu_min = np.ceil(np.sqrt(max(1+(nr_verts-12)/10, 1)))
            nu = max(nu, nu_min)
            nu = int(nu)
            
        # Subdividing  
        if nu>1:
            (vertices,faces) = self.subdivide_mesh(vertices, faces, nu)
            vertices = vertices/np.sqrt(np.sum(vertices**2, axis=1, keepdims=True))

        return(vertices, faces)

    def icosahedron(self):
        '''' Regular unit icosahedron. '''
        
        # 12 principal directions in 3D space: points on an unit icosahedron
        phi = (1+np.sqrt(5))/2    
        vertices = np.array([
            [0, 1, phi], [0,-1, phi], [1, phi, 0],
            [-1, phi, 0], [phi, 0, 1], [-phi, 0, 1]])/np.sqrt(1+phi**2)
        vertices = np.r_[vertices,-vertices]
        
        # 20 faces
        faces = np.array([
            [0,5,1], [0,3,5], [0,2,3], [0,4,2], [0,1,4], 
            [1,5,8], [5,3,10], [3,2,7], [2,4,11], [4,1,9], 
            [7,11,6], [11,9,6], [9,8,6], [8,10,6], [10,7,6], 
            [2,11,7], [4,9,11], [1,8,9], [5,10,8], [3,7,10]], dtype=int)    
        
        return (vertices, faces)


    def subdivide_mesh(self, vertices, faces, nu):
        '''
        Subdivides mesh by adding vertices on mesh edges and faces. Each edge 
        will be divided in nu segments. (For example, for nu=2 one vertex is added  
        on each mesh edge, for nu=3 two vertices are added on each mesh edge and 
        one vertex is added on each face.) If V and F are number of mesh vertices
        and number of mesh faces for the input mesh, the subdivided mesh contains 
        V + F*(nu+1)*(nu-1)/2 vertices and F*nu^2 faces.
        
        Parameters
        ----------
        vertices : vertex list, numpy array of shape (V,3) 
        faces : face list, numby array of shape (F,3). Zero indexed.
        nu : subdivision frequency, integer (larger than 1 to make a change).
        
        Returns
        -------
        subvertices : vertex list, numpy array of shape (V + F*(nu+1)*(nu-1)/2, 3)
        subfaces : face list, numpy array of shape (F*n**2, 3)
        
        Author: vand at dtu.dk, 8.12.2017. Translated to python 6.4.2021
        
        '''
            
        edges = np.r_[faces[:,:-1], faces[:,1:],faces[:,[0,2]]]
        edges = np.unique(np.sort(edges, axis=1),axis=0)
        F = faces.shape[0]
        V = vertices.shape[0]
        E = edges.shape[0] 
        subfaces = np.empty((F*nu**2, 3), dtype = int)
        subvertices = np.empty((V+E*(nu-1)+F*(nu-1)*(nu-2)//2, 3))
                            
        subvertices[:V] = vertices
        
        # Dictionary for accessing edge index from indices of edge vertices.
        edge_indices = dict()
        for i in range(V):
            edge_indices[i] = dict()
        for i in range(E):
            edge_indices[edges[i,0]][edges[i,1]] = i
            edge_indices[edges[i,1]][edges[i,0]] = -i
            
        template = self.faces_template(nu)
        ordering = self.vertex_ordering(nu)
        reordered_template = ordering[template]
        
        # At this point, we have V vertices, and now we add (nu-1) vertex per edge
        # (on-edge vertices).
        w = np.arange(1,nu)/nu # interpolation weights
        for e in range(E):
            edge = edges[e]
            for k in range(nu-1):
                subvertices[V+e*(nu-1)+k] = (w[-1-k] * vertices[edge[0]] 
                                            + w[k] * vertices[edge[1]])
      
        # At this point we have E(nu-1)+V vertices, and we add (nu-1)*(nu-2)/2 
        # vertices per face (on-face vertices).
        r = np.arange(nu-1)
        for f in range(F):
            # First, fixing connectivity. We get hold of the indices of all
            # vertices invoved in this subface: original, on-edges and on-faces.
            T = np.arange(f*(nu-1)*(nu-2)//2+E*(nu-1)+V, 
                          (f+1)*(nu-1)*(nu-2)//2+E*(nu-1)+V) # will be added
            eAB = edge_indices[faces[f,0]][faces[f,1]] 
            eAC = edge_indices[faces[f,0]][faces[f,2]] 
            eBC = edge_indices[faces[f,1]][faces[f,2]] 
            AB = self.reverse(abs(eAB)*(nu-1)+V+r, eAB<0) # already added
            AC = self.reverse(abs(eAC)*(nu-1)+V+r, eAC<0) # already added
            BC = self.reverse(abs(eBC)*(nu-1)+V+r, eBC<0) # already added
            VEF = np.r_[faces[f], AB, AC, BC, T]
            subfaces[f*nu**2:(f+1)*nu**2, :] = VEF[reordered_template]
            # Now geometry, computing positions of face vertices.
            subvertices[T,:] = self.inside_points(subvertices[AB,:],subvertices[AC,:])
        
        return (subvertices, subfaces)
    

    def reverse(self, vector, flag): 
        '''' For reversing the direction of an edge. ''' 
        
        if flag:
            vector = vector[::-1]
        return(vector)

    
    def faces_template(self, nu):
        '''
        Template for linking subfaces                  0
        in a subdivision of a face.                   / \
        Returns faces with vertex                    1---2
        indexing given by reading order             / \ / \
        (as illustratated).                        3---4---5
                                                  / \ / \ / \
                                                6---7---8---9    
                                                / \ / \ / \ / \ 
                                              10--11--12--13--14 
        '''
      
        faces = []
        # looping in layers of triangles
        for i in range(nu): 
            vertex0 = i*(i+1)//2
            skip = i+1      
            for j in range(i): # adding pairs of triangles, will not run for i==0
                faces.append([j+vertex0, j+vertex0+skip, j+vertex0+skip+1])
                faces.append([j+vertex0, j+vertex0+skip+1, j+vertex0+1])
            # adding the last (unpaired, rightmost) triangle
            faces.append([i+vertex0, i+vertex0+skip, i+vertex0+skip+1])
            
        return (np.array(faces))


    def vertex_ordering(self, nu):
        ''' 
        Permutation for ordering of                    0
        face vertices which transformes               / \
        reading-order indexing into indexing         3---6
        first corners vertices, then on-edges       / \ / \
        vertices, and then on-face vertices        4---12--7
        (as illustrated).                         / \ / \ / \
                                                5---13--14--8
                                                / \ / \ / \ / \ 
                                              1---9--10--11---2 
        '''
        
        left = [j for j in range(3, nu+2)]
        right = [j for j in range(nu+2, 2*nu+1)]
        bottom = [j for j in range(2*nu+1, 3*nu)]
        inside = [j for j in range(3*nu,(nu+1)*(nu+2)//2)]
        
        o = [0] # topmost corner
        for i in range(nu-1):
            o.append(left[i])
            o = o + inside[i*(i-1)//2:i*(i+1)//2]
            o.append(right[i])
        o = o + [1] + bottom + [2]
            
        return(np.array(o))
    
    def inside_points(self, vAB,vAC):
        '''  
        Returns coordinates of the inside                 .
        (on-face) vertices (marked by star)              / \
        for subdivision of the face ABC when         vAB0---vAC0
        given coordinates of the on-edge               / \ / \
        vertices  AB[i] and AC[i].                 vAB1---*---vAC1
                                                    / \ / \ / \
                                                vAB2---*---*---vAC2
                                                  / \ / \ / \ / \
                                                  .---.---.---.---. 
        '''
      
        v = []
        for i in range(1,vAB.shape[0]):
            w = np.arange(1,i+1)/(i+1)
            for k in range(i):
                v.append(w[-1-k]*vAB[i,:] + w[k]*vAC[i,:])
        
        return(np.array(v).reshape(-1,3)) # reshape needed for empty return

    @property
    def num_directions(self):
        """Getter for the num_directions"""
        return self._num_directions

    @num_directions.setter
    def num_directions(self, num_directions: int):
        """Sets the number of directions and updates the directions."""
        self._num_directions = num_directions
        vertices, _ = icosphere.icosphere(nr_verts=self._num_directions)
        self.directions = torch.from_numpy(vertices).float()

    def generate_direction_samples(self, num_directions=None, apply_random_rotation=None) -> RaySamples:
        # generate N random rotations
        if num_directions is None:
            directions = self.directions
        else:
            vertices, _ = icosphere.icosphere(nr_verts=num_directions)
            directions = torch.from_numpy(vertices).float()
            
        if (apply_random_rotation is None and self.apply_random_rotation) or apply_random_rotation is True:
            R = torch.from_numpy(Rotation.random(1).as_matrix())[0].float()
            directions = directions @ R

        if self.remove_lower_hemisphere:
            directions = directions[directions[:, 2] > 0]

        origins = torch.zeros_like(directions)

        ray_samples = RaySamples(frustums=Frustums(origins=origins,
                                                   directions=directions,
                                                   starts=torch.zeros_like(directions[:, 0]),
                                                   ends=torch.ones_like(directions[:, 0]),
                                                   pixel_area=torch.ones_like(directions[:, 0]))
        )

        return ray_samples


# Field related configs
@dataclass
class EquirectangularSamplerConfig(IlluminationSamplerConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: EquirectangularSampler)
    """target class to instantiate"""
    width: int = 256
    """width of the equirectangular image"""
    apply_random_rotation: bool = False
    """apply random rotation to the directions"""
    remove_lower_hemisphere: bool = False
    """remove lower hemisphere"""


class EquirectangularSampler(IlluminationSampler):
    """For sampling directions from an icosahedron."""

    def __init__(
        self,
        config: EquirectangularSamplerConfig,
    ):
        super().__init__(config)
        self._width = config.width
        self.height = self._width // 2
        self.apply_random_rotation = config.apply_random_rotation
        self.remove_lower_hemisphere = config.remove_lower_hemisphere

        cx = torch.tensor(self.width // 2, dtype=torch.float32).repeat(1)
        cy = torch.tensor(self.height // 2, dtype=torch.float32).repeat(1)
        fx = torch.tensor(self.height, dtype=torch.float32).repeat(1)
        fy = torch.tensor(self.height, dtype=torch.float32).repeat(1)

        c2w = torch.tensor([[[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0]]], dtype=torch.float32).repeat(1, 1, 1) # From nerfstudio world to camera
                            
        self.camera = Cameras(fx=fx, fy=fy, cx=cx, cy=cy, camera_to_worlds=c2w, camera_type=CameraType.EQUIRECTANGULAR)

    @property
    def width(self):
        """Getter for the width"""
        return self._width

    @width.setter
    def width(self, width: int):
        """Sets the number of directions and updates the directions."""
        self._width = width
        # update cx, cy, fx, fy
        cx = torch.tensor(self.width // 2, dtype=torch.float32).repeat(1)
        cy = torch.tensor(self.height // 2, dtype=torch.float32).repeat(1)
        fx = torch.tensor(self.height, dtype=torch.float32).repeat(1)
        fy = torch.tensor(self.height, dtype=torch.float32).repeat(1)

        self.camera.cx = cx
        self.camera.cy = cy
        self.camera.fx = fx
        self.camera.fy = fy

    def generate_direction_samples(self, num_directions=None, apply_random_rotation=None) -> RaySamples:
        # generate N random rotations
        ray_bundle = self.camera.generate_rays(camera_indices=0, keep_shape=False)

        directions = ray_bundle.directions
        camera_indices = ray_bundle.camera_indices

        ray_samples = RaySamples(frustums=Frustums(origins=torch.zeros_like(directions),
                                                   directions=directions,
                                                   starts=torch.zeros_like(directions[:, 0]),
                                                   ends=torch.ones_like(directions[:, 0]),
                                                   pixel_area=torch.ones_like(directions[:, 0])),
                                 camera_indices=camera_indices
        )

        return ray_samples
