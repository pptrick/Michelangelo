import os
import trimesh
import numpy as np

class MeshSampler:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh = None
        try:
            if not os.path.exists(self.mesh_path):
                raise FileNotFoundError(f"{self.mesh_path} not found!")
            self.mesh = trimesh.load_mesh(self.mesh_path)
            
            # filter scene or mesh
            if isinstance(self.mesh, trimesh.Scene):
                self.mesh = trimesh.util.concatenate([m for m in self.mesh.geometry.values()])
        except Exception as e:
            print(f"Error when initializing mesh sampler: {e}")
            
    def sample(self, n_points=4096, save_path=None):
        if self.mesh is not None:
            points, face_indices = trimesh.sample.sample_surface_even(self.mesh, n_points)
            normals = self.mesh.face_normals[face_indices]
            if save_path is not None and isinstance(save_path, str):
                save_path = save_path if save_path.endswith(".npz") else save_path + ".npz" 
                np.savez_compressed(save_path, points=points, normals=normals)
            else:
                surface = np.concatenate([points, normals], axis=-1)
                return surface