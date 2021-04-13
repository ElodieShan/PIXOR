import torch
import numpy as np
from physical_car.load import *
from physical_car.utils import *

def shoot(mesh, rays):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    D_p = torch.stack([rays[:, 0].cos() * rays[:, 1].cos(),  rays[:, 0].sin() * rays[:, 1].cos(), rays[:, 1].sin()], dim=1)
    A = verts[faces[:, 0]]
    E1 = verts[faces[:, 1]] - A
    E2 = verts[faces[:, 2]] - A
    D = D_p.unsqueeze(1)
    A.unsqueeze_(0)
    E1.unsqueeze_(0)
    E2.unsqueeze_(0)
    D, A, E1, E2 = torch.broadcast_tensors(D, A, E1, E2)
    dot = lambda a, b: (a*b).sum(-1)
    denom = dot(D ,(E1.cross(E2))) + 1e-10
    t = dot(A, E1.cross(E2)) / denom
    u = dot(A, D.cross(E2)) / denom
    v = dot(A, E1.cross(D)) / denom
    w = 1 - u - v
    if_shoot = (u>=0).logical_and(v>=0).logical_and(w>=0)
    if if_shoot.sum().item() == 0:
        return None
    
    shoot_rays = if_shoot.sum(1) > 0
    if_shoot = if_shoot[shoot_rays]
    t = t[shoot_rays]
    t[if_shoot.logical_not()] = 1e20
    value, ind = t.min(1)
    new_pts = D_p[shoot_rays] * value.unsqueeze(1)
    return new_pts


def adv_bbox(bb, size=[1.0, 1.0, 1.0]):
    adv_bb = bb.new_zeros(7)
    adv_bb[:2] = bb[:2]
    adv_bb[2] = bb[2] + bb[5]/2 + size[2] / 2 - 0.2
    adv_bb[3] = size[0]
    adv_bb[4] = size[1]
    adv_bb[5] = size[2]
    adv_bb[6] = bb[6]
    return adv_bb

def sample_pc(mesh, bb, adv_bb, dalpha=0.4/180 * np.pi, dtheta=0.17/180*np.pi, device=None):
    physical_transforms(mesh, vector=adv_bb[:3], angle=bb[6])
#     physical_transforms(mesh, vector=adv_bb[:3])
#     physical_transforms(mesh, angle=bb[6], center=adv_bb[:3])
    # the distance
    dis2 = adv_bb[:2].norm()
    dis3 = adv_bb[:3].norm()

    # the center of the onject
    theta0 = torch.atan2(adv_bb[1], adv_bb[0])
    alpha0 = torch.asin(adv_bb[2]/dis3)

    # the half max degree change of the object
    theta_hmd = torch.atan(adv_bb[3:5].norm() / 2 / dis2)
    alpha_hmd = torch.atan(adv_bb[5] /2 / dis3)

#     alpha_calib = torch.asin(pts[:, 3] / dis3).max()
    alpha_calib = alpha0
    upper = lambda a, c, d: np.ceil((a - c) / d) * d + c
    alpha_sample = torch.arange((alpha0 - alpha_hmd).item(), (alpha0 + alpha_hmd).item(), dalpha, device=device)
    theta_sample = torch.arange((theta0 - theta_hmd).item(), (theta0 + theta_hmd).item(), dtheta, device=device)
    rays = torch.stack(torch.meshgrid(theta_sample, alpha_sample), dim=2).reshape(-1, 2)
    new_pts = shoot(mesh, rays)
    return new_pts

def sample_pc_test(mesh, adv_bb, dalpha=0.4/180 * np.pi, dtheta=0.17/180*np.pi, device=None):
    physical_transforms(mesh, vector=adv_bb[:3], angle=adv_bb[6])
#     physical_transforms(mesh, vector=adv_bb[:3])
#     physical_transforms(mesh, angle=bb[6], center=adv_bb[:3])
    # the distance
    dis2 = adv_bb[:2].norm()
    dis3 = adv_bb[:3].norm()

    # the center of the onject
    theta0 = torch.atan2(adv_bb[1], adv_bb[0])
    alpha0 = torch.asin(adv_bb[2]/dis3)

    # the half max degree change of the object
    # theta_hmd = torch.atan(adv_bb[3:5].norm() / 2 / dis2)
    # alpha_hmd = torch.atan(adv_bb[5] /2 / dis3)
    theta_hmd = 0
    alpha_hmd = 0
#     alpha_calib = torch.asin(pts[:, 3] / dis3).max()
    alpha_calib = alpha0
    upper = lambda a, c, d: np.ceil((a - c) / d) * d + c
    alpha_sample = torch.arange((alpha0 - alpha_hmd).item(), (alpha0 + alpha_hmd).item(), dalpha, device=device)
    theta_sample = torch.arange((theta0 - theta_hmd).item(), (theta0 + theta_hmd).item(), dtheta, device=device)
    rays = torch.stack(torch.meshgrid(theta_sample, alpha_sample), dim=2).reshape(-1, 2)
    new_pts = shoot(mesh, rays)
    return new_pts