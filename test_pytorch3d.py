import os
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_normal_consistency,
    mesh_laplacian_smoothing
)
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
# from alfred.dl.torch.common import device
import open3d as o3d

# from physical_car.load import *
# from physical_car.utils import *
import physical_car.visualize_utils as V
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
import matplotlib.colors as mcolors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# trg_obj = os.path.join('./data', 'PIXOR/physical_car/car/AUDIA3.obj')
verts, faces, aux = load_obj('physical_car/car/AUDIA3.obj')
print('verts: ', verts)
print('faces: ', faces)
print('aux: ', aux)


face_idx = faces.verts_idx.to(device)
verts = verts.to(device)
# center = verts.mean(0)
# verts = verts - center
# scale = max(verts.abs().max(0)[0])
# verts = verts / scale

trg_mesh = Meshes(verts=[verts], faces=[face_idx])
src_mesh = ico_sphere(4, device)

# we can print verts as well, using open3d
verts_points = verts.clone().detach().cpu().numpy()
# print(verts_points)
pcobj = o3d.geometry.PointCloud()
pcobj.points = o3d.utility.Vector3dVector(verts_points)
# o3d.visualization.draw_geometries([pcobj])
# o3d.io.write_point_cloud('physical_car/car/AUDIA3.ply',pcobj)
# # ​

ax = V.plot_points_Vector3dVector(verts, axis='on', facecolor='w')
# plt.show()
# save_path = os.path.join(verts_res_save_dir, 'mesh', ('%s_mesh.png' % i))
plt.savefig('physical_car/car/AUDIA3.png')