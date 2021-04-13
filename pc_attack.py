
import os
import copy
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from physical_car.load import *
from physical_car.utils import *
import physical_car.visualize_utils as V
from simulate.lidar_shoot import *
from load_PIXOR import LoadPIXOR
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.image import imread
import matplotlib.colors as mcolors

from visualization_utils.draw_results import *
import pytorch3d as p3d
import pytorch3d.utils
import pytorch3d.io as p3dio
from pytorch3d.structures import Meshes
import pytorch3d.loss

from evaluate_model import *
from detector import *

from load_PIXOR import *
# from loss import CustomLoss
# from datagen import get_data_loader
# from model import PIXOR
# from utils import get_model_name, load_config, get_logger, plot_bev, plot_label_map, plot_pr_curve, get_bev
# from postprocess import filter_pred, compute_matches, compute_ap
# from main import *

# from utils_temporary_change_name.meshes import ico_sphere

os.environ['QT_QPA_PLATFORM']='offscreen'

x_MIN = 0.0
x_MAX = 70.0
y_MIN =-40.0
y_MAX = 40.0
z_MIN = -2.5
z_MAX = 1
x_DIVISION = 0.1
y_DIVISION = 0.1
z_DIVISION = 0.1
x_INDEX_MAX=700
y_INDEX_MAX=800
z_INDEX_MAX=35

def data_gen():
    while True:
        for data_dict in train_loader:
            yield data_dict

def getXYZ(s):
    # s = ((s - s.new([x_MIN, y_MIN, z_MIN])) / s.new([x_DIVISION, y_DIVISION, z_DIVISION]))
    s[...,0] = (x_INDEX_MAX - (s[...,0] - x_MIN)/x_DIVISION)
    s[...,1] = (-s[...,1] - y_MIN)/y_DIVISION
    s[...,2] = (s[...,2] - z_MIN)/z_DIVISION
    inds = (s[..., 0] >= 0).logical_and(s[..., 1] >= 0).logical_and(s[..., 2] >= 0)
    inds = inds.logical_and(s[..., 0] < x_INDEX_MAX).logical_and(s[..., 1] < y_INDEX_MAX).logical_and(s[..., 2] < z_INDEX_MAX)
    s = s[inds]
    return s

def mapping_fun(inputs, new_pts, size=1, alpha=1):
    device = inputs.device
    inputs = inputs.permute([1,2,0]) # shape [36, 700, 800] to [700, 800, 36]
    # method1
#     new_cord = getXYZ(new_pts).unsqueeze(1)
#     x = torch.arange(0.0, size*2, device=device) - size + 0.5
#     y = torch.arange(0.0, size*2, device=device) - size + 0.5
#     z = torch.arange(0.0, size*2, device=device) - size + 0.5
#     nb_cord = torch.stack(torch.meshgrid(x, y, z), axis=-1).view(-1, 3)
#     nb_cord = new_cord + nb_cord
#     nb_cord = nb_cord.long()
#     loc_cord = (new_cord - nb_cord) - 0.5
#     dist = loc_cord.abs().sum(-1) # define distance
#     w = (dist * alpha).exp()
#     w = w / w.sum(-1, keepdim=True)
    
#     inds = (nb_cord[..., 0] >= 0).logical_and(nb_cord[..., 1] >= 0).logical_and(nb_cord[..., 2] >= 0)
#     inds = inds.logical_and(nb_cord[..., 0] < x_INDEX_MAX).logical_and(nb_cord[..., 1] < y_INDEX_MAX).logical_and(nb_cord[..., 2] < z_INDEX_MAX)
#     nb_cord = nb_cord[inds].view(-1, 3)
#     w = w[inds].view(-1)
# #     print(inputs.dtype)
# #     print(nb_cord.dtype)
# #     print(w.dtype)
# #     inputs.index_put_((nb_cord[:, 2], nb_cord[:, 1], nb_cord[:, 0]), w, True)
#     inputs[nb_cord[:, 2], nb_cord[:, 1], nb_cord[:, 0]] += w
# #     for i in range(len(w)):
# #         inputs[nb_cord[i, 2], nb_cord[i, 1], nb_cord[i, 0]] += w[i]

    # method2
    # new_cord = getXYZ(new_pts).flip(-1)
    new_cord = getXYZ(new_pts)

    if new_cord.shape[0] > 0:
        x = new_cord[..., 0]
        y = new_cord[..., 1]
        z = new_cord[..., 2]
        x_min = max((x.min() - size + 0.5).long(), 0)
        y_min = max((y.min() - size + 0.5).long(), 0)
        z_min = max((z.min() - size + 0.5).long(), 0)
        x_max = min((x.max() + size + 0.5).long(), inputs.shape[0])
        y_max = min((y.max() + size + 0.5).long(), inputs.shape[1])
        z_max = min((z.max() + size + 0.5).long(), inputs.shape[2])
        if x_max>x_min and y_max>y_min and z_max>z_min:
            xl = torch.arange(x_min, x_max, device=device) + 0.5
            yl = torch.arange(y_min, y_max, device=device) + 0.5
            zl = torch.arange(z_min, z_max, device=device) + 0.5
            nb_cord = torch.stack(torch.meshgrid(xl, yl, zl), axis=-1).unsqueeze(-2)
            dis = (new_cord.view(1, 1, 1, -1, 3) - nb_cord).abs()
            inds = (dis[..., 0] < size).logical_and(dis[..., 1] < size).logical_and(dis[..., 2] < size)
            w = (-dis.sum(-1)*alpha).exp()
            w = w * inds
            w = w / w.sum([0, 1, 2])
            w = w.sum(-1)
            inputs[x_min:x_max, y_min:y_max, z_min:z_max] += w
            inputs.data[x_min:x_max, y_min:y_max, z_min:z_max] -= w
            
        new_cord = new_cord.long()
        inputs.data.index_put_([new_cord[:, 0], new_cord[:, 1], new_cord[:, 2]], torch.tensor(1., device=device))

    inputs = inputs.permute([2,0,1]) # shape [700, 800, 36] to [36, 700, 800] 
    return inputs, new_cord.shape[0]

def get_data_dict(data_loader, file_id):
    # Load input data
    # file_id = i%len(data_loader.dataset)
    voxel_point_cloud, labels, calib, training_labels = data_loader.dataset[file_id]

    gt_boxes_lidar = []
    for label in labels:
        if label.box3d_lidar is not None:
            gt_boxes_lidar.append(label.box3d_lidar)
    # if len(gt_boxes_lidar) <= 0:
    #     return None, None

    data_dict = {
        "id": file_id,
        "input": voxel_point_cloud,
        "gt_boxes": torch.from_numpy(np.array(gt_boxes_lidar)).to(device),
        "cls_targets": training_labels[:, :, -1],
        "loc_targets": training_labels[:, :, :-1],
    }
    return data_dict, labels, calib


def attack_model(model, data_loader, device, res_save_dir=None):
    root = "/home/elodie"
    suffix = 'try_s1_a1'

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join(root, 'data/runs', TIMESTAMP + '_' + suffix))

    # Attack eesh result save dir
    verts_res_save_dir = os.path.join(res_save_dir,TIMESTAMP)
    if not os.path.exists(verts_res_save_dir):
        os.makedirs(verts_res_save_dir)
        os.makedirs(os.path.join(verts_res_save_dir,"mesh"))
        os.makedirs(os.path.join(verts_res_save_dir,"verts"))

    # start optimization loop
    mesh0 = p3d.utils.ico_sphere(2, device)
    print("mesh0:",mesh0)
    physical_transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))
    # mesh0 = ico_sphere(2, device)
    # transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))

    deform_verts = mesh0.verts_packed().new_zeros(mesh0.verts_packed().shape).requires_grad_()
    optimizer = torch.optim.Adam([deform_verts], lr=0.001)

    # Niter = 100
    Niter = 37120
    plot_period = 200
    res_save_period = plot_period * 10

    # w_chamfer = 1.0
    # w_adv = 1.0
    # w_edge = 1.0
    # w_normal = 0.01
    w_laplacian = 0.001

    bound = deform_verts.new([0.1, 0.1, 0])

    adv_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    losses = []

    data_iter = data_gen()

    deform_verts_result = []

    for i in tqdm(range(Niter)):
        # Initialize optimizer
        optimizer.zero_grad()

        # Load input data
        file_id = i%len(data_loader.dataset)
        voxel_point_cloud, labels, calib, training_labels = data_loader.dataset[file_id]

        gt_boxes_lidar = []
        for label in labels:
            if label.box3d_lidar is not None:
                gt_boxes_lidar.append(label.box3d_lidar)
        if len(gt_boxes_lidar) <= 0:
            continue

        data_dict = {
            "id": file_id,
            "input": voxel_point_cloud,
            "gt_boxes": torch.from_numpy(np.array(gt_boxes_lidar)).to(device),
            "cls_targets": training_labels[:, :, -1],
            "loc_targets": training_labels[:, :, :-1],
        }
        # Add attack mesh
        new_pts_num_all = 0
        for bb in data_dict['gt_boxes']:
            mesh = mesh0.offset_verts(torch.tanh(deform_verts) * bound)
            adv_bb = adv_bbox(bb)
            # print("bb:",bb)
            new_pts = sample_pc(mesh, bb, adv_bb, device=device)
            # print("new_pts:",new_pts)
            if new_pts is not None:
                data_dict["input"], new_pts_num = mapping_fun(data_dict["input"], new_pts, size=1, alpha=1)
                new_pts_num_all += new_pts_num
            else:
                warnings.warn("Warning Lidar shoot nothing")
        #     if new_pts is not None:
        #         new_pts[:,0] = (x_INDEX_MAX - (new_pts[:,0] - x_MIN)/x_DIVISION)
        #         new_pts[:,1] = (-new_pts[:,1] - y_MIN)/y_DIVISION
        #         new_pts[:,2] = (new_pts[:,2] - z_MIN)/z_DIVISION
        #         new_pts_index = new_pts.long()
        #         # print("new_pts_index:",new_pts_index)
        #         index_mask = torch.vstack((new_pts_index[:,0]<x_INDEX_MAX, new_pts_index[:,1]<y_INDEX_MAX, new_pts_index[:,2]<z_INDEX_MAX)).all(dim=0)
        #         if index_mask.sum() > 0:
        #             new_pts_index = new_pts_index[index_mask][:, [2, 0, 1]] # Z\X\Y torch.Size([36, 700, 800])
        #             new_pts_index_num = new_pts_index.shape[0]
        #             new_pts_index=new_pts_index.permute(1,0)
        #             data_dict['voxel_point_cloud'].index_put_(tuple(new_pts_index), torch.ones(new_pts_index_num).to(device))
        #         new_pts_num += index_mask.sum()
        #     else:
        #         warnings.warn("Warning Lidar shoot nothing")
        if new_pts_num_all ==0:
            warnings.warn("Mesh remains None!")
            # print("Mesh remains None! Skip..")
            continue

        pred_dicts = PIXOR(data_dict)
        cls_pred = pred_dicts["batch_predictions"][0][...,-1]
        pred_scores = cls_pred[pred_dicts['cls_targets'] == 1]

        loss_adv = - (1 - pred_scores).log().mean()
        loss_laplacian = loss_adv.new_zeros(1)
        # loss_laplacian = p3d.loss.mesh_laplacian_smoothing(mesh, method="uniform")
        loss = loss_adv + loss_laplacian * w_laplacian

        writer.add_scalar('loss/loss_total', loss.item(), i)
        #     writer.add_scalar('loss/loss_adv', loss_adv.item(), i)
        #     writer.add_scalar('loss/loss_edge', loss_edge.item(), i)
        #     writer.add_scalar('loss/loss_normal', loss_normal.item(), i)
        #     writer.add_scalar('loss/loss_laplacian', loss_laplacian.item(), i)

        # ------------- Plot mesh
        if i % plot_period == 0:
            print('i: {}, loss: {}'.format(i, loss))


            ax = V.plot_points(mesh0.offset_verts(torch.tanh(deform_verts) * bound).verts_packed(), axis='on', facecolor='w')
            # plt.show()
            save_path = os.path.join(verts_res_save_dir, 'mesh', ('%s_mesh.png' % i))
            plt.savefig(save_path)
            fig = ax.get_figure()
            writer.add_figure('mesh', fig, i)

            res_save_path = os.path.join(verts_res_save_dir, 'verts',('%s_verts.pkl' % i))

            with open(res_save_path, 'wb') as f:
                pickle.dump(deform_verts.detach(), f)

            #         w, h = [374, 384]
            #         fig_np = fig.canvas.tostring_rgb()
            #         fig_np = np.frombuffer(fig_np, dtype=np.uint8).reshape(w, h, 3).transpose(2,0,1)
            #         writer.add_image('mesh', fig_np, i)
            # scene = V.plot_scene(data_dict, pred_dict)
            # # plt.show()
            # # path = 'Figures/' + str
            # save_path = os.path.join('Figures', ('%s_result.png' % i))
            # plt.savefig(save_path)
            #         fig = ax.get_figure()
            #         writer.add_figure('scene', fig, i)

            writer.flush()

        # Optimization step
        loss.backward()
        # print(deform_verts.grad)
        optimizer.step()
        torch.cuda.empty_cache()

    # print("exp_name:",exp_name)
    # res_save_path = os.path.join('logs',exp_name, ('verts_%s.pkl' % TIMESTAMP))
    res_save_path = os.path.join(verts_res_save_dir, ('verts_%s.pkl' % TIMESTAMP))

    with open(res_save_path, 'wb') as f:
        pickle.dump(deform_verts.detach(), f)


def plot_bev_res_image(data_dict, labels, calib, img_save_path, id):

        ###################
        # display results #
        ###################

        # set colors
        ground_truth_color = (80, 127, 255)
        prediction_color = (255, 127, 80)

        # get point cloud as numpy array
        point_cloud = data_dict["input"].detach().cpu().numpy().transpose((1, 2, 0))
        final_box_predictions = data_dict["final_box_predictions"]
        # draw BEV image
        bev_image = kitti_utils.draw_bev_image(point_cloud)

        # display ground truth bounding boxes on BEV image and camera image
        for label in labels:
            # only consider annotations for class "Car"
            if label.type == 'Car':
                # compute corners of the bounding box
                bbox_corners_image_coord, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P)
                # display bounding box in BEV image
                bev_img = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord, color=ground_truth_color)
                # display bounding box in camera image
                # if bbox_corners_image_coord is not None:
                    # camera_image = kitti_utils.draw_projected_box_3d(camera_image, bbox_corners_image_coord, color=ground_truth_color)

        # display predicted bounding boxes on BEV image and camera image
        if final_box_predictions is not None:
            print("final_box_predictions:",final_box_predictions.shape)
            final_box_predictions = final_box_predictions
            for prediction in final_box_predictions:
                print("prediction:",prediction)
                bbox_corners_camera_coord = np.reshape(prediction[2:10], (2, 4)).T
                # create 3D bounding box coordinates from BEV coordinates. Place all bounding boxes on the ground and
                # choose a height of 1.5m
                bbox_corners_camera_coord = np.tile(bbox_corners_camera_coord, (2, 1))
                bbox_y_camera_coord = np.array([[0., 0., 0., 0., 1.65, 1.65, 1.65, 1.65]]).T
                bbox_corners_camera_coord = np.hstack((bbox_corners_camera_coord, bbox_y_camera_coord))
                switch_indices = np.argsort([0, 2, 1])
                bbox_corners_camera_coord = bbox_corners_camera_coord[:, switch_indices]
                bbox_corners_image_coord = kitti_utils.project_to_image(bbox_corners_camera_coord, calib.P)

                # display bounding box with confidence score in BEV image
                bev_img = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord, color=prediction_color, confidence_score=prediction[1])
                # display bounding box in camera image
                # if bbox_corners_image_coord is not None:
                    # camera_image = kitti_utils.draw_projected_box_3d(camera_image, bbox_corners_image_coord, color=prediction_color)

        # display legend on BEV Image
        bev_image = show_legend(bev_image, ground_truth_color, prediction_color, id)

        # show images
        # cv2.imshow('BEV Image', bev_image)
        # cv2.imshow('Camera Image', camera_image)
        # cv2.waitKey()

        # save image
        print('Index: ', id)
        
        cv2.imwrite(img_save_path, bev_image)

############
# show attack effect  #
############

def experiment_attack(model, data_loader, device, deform_verts, plot=True, res_save_dir=None, images_num=10):

    # print("mesh_res_pkl:",mesh_res_pkl)
    # with open(mesh_res_pkl, 'rb') as f:
    #     mesh_res = pickle.load(f)
    # print("mesh_res:",mesh_res)
    # Train Set

    # Attack eesh result save dir
    img_save_dir = res_save_dir
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    if not os.path.exists(os.path.join(img_save_dir,"ori")):
        os.makedirs(os.path.join(img_save_dir,"ori"))
    if not os.path.exists(os.path.join(img_save_dir,"attack")):
        os.makedirs(os.path.join(img_save_dir,"attack"))


    # start optimization loop
    mesh0 = p3d.utils.ico_sphere(2, device)
    print("mesh0:",mesh0)
    physical_transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))
    bound = deform_verts.new([0.1, 0.1, 0])
    ids = np.arange(0, images_num)
    for id in ids:
        data_dict, labels, calib = get_data_dict(data_loader, id%len(data_loader.dataset))
        # Ori detection results
        pred_dicts_ori = PIXOR(data_dict)
        img_save_path = os.path.join(img_save_dir,"ori",'detection_id_{:d}.png'.format(id))
        plot_bev_res_image(pred_dicts_ori, labels, calib, img_save_path, id)
        
        # Add attack mesh
        new_pts_num_all = 0
        for bb in data_dict['gt_boxes']:
            mesh = mesh0.offset_verts(torch.tanh(deform_verts) * bound)
            adv_bb = adv_bbox(bb)
            # print("bb:",bb)
            new_pts = sample_pc(mesh, bb, adv_bb, device=device)
            # print("new_pts:",new_pts)
            if new_pts is not None:
                data_dict["input"], new_pts_num = mapping_fun(data_dict["input"], new_pts, size=1, alpha=1)
                new_pts_num_all += new_pts_num
            else:
                warnings.warn("Warning Lidar shoot nothing")
        # Ori detection results
        pred_dicts_attack = PIXOR(data_dict)
        img_save_path_attack = os.path.join(img_save_dir,"attack",'detection_id_{:d}.png'.format(id))
        plot_bev_res_image(pred_dicts_attack, labels, calib, img_save_path_attack, id)
        
    # print("Validation mAP", val_metrics['AP'])
    # print("Net Fwd Pass Time on average {:.4f}s".format(val_metrics['Forward Pass Time']))
    # print("Nms Time on average {:.4f}s".format(val_metrics['Postprocess Time']))

    # fig_name = "PRCurve_val_IOU7-attack" + config['name']
    # legend = "AP={:.1%} @IOU=0.7-attack".format(val_metrics['AP'])

    # # legend = "AP={:.1%} @IOU=0.5".format(val_metrics['AP'])
    # plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)


if __name__ == "__main__":
    # test mode or train mode
    mode = "test"
    mesh_res_pkl = "output_models/20210403_ImageSets55_Ori_Model/attack/2021-04-08T13-36-22/verts/200_verts.pkl"
    images_num = 20
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)
    # Load PIXOR Detector 
    n_epochs_trained = 17
    model_dir = "output_models/20210403_ImageSets55_Ori_Model"
    use_voxelize_density = False
    use_ImageSets = True

    # create PIXOR model
    PIXOR = LoadPIXOR(device, n_epochs_trained=n_epochs_trained, model_dir=model_dir, 
                use_voxelize_density=use_voxelize_density, use_ImageSets=use_ImageSets)
    
    # create data loader
    root_dir = 'Data/'
    batch_size = 1

    if mode == "train":
        data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device,  \
                use_voxelize_density=use_voxelize_density, use_ImageSets=use_ImageSets, \
                attack_training_mode=True)
        print("Successfully get data_loader.. ")
        res_save_dir = model_dir + "/attack/"
        attack_model(PIXOR, data_loader, device, res_save_dir=res_save_dir)
    elif mode == "test":
        data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device,  \
                use_voxelize_density=use_voxelize_density, use_ImageSets=use_ImageSets, \
                attack_training_mode=True,test_set=True)
        print("Successfully get data_loader.. ")

        # mesh_res_pkl = "logs/attack/test3/2021-03-25T14-11-14/verts_9000.pkl"
        # experiment_attack(exp_name, device, mesh_res_pkl=mesh_res_pkl, eval_range='all', plot=True)

        print("mesh_res_pkl:",mesh_res_pkl)
        with open(mesh_res_pkl, 'rb') as f:
            deform_verts = pickle.load(f)
        # print("mesh_res:",mesh_res)
        mesh_dir = mesh_res_pkl.split('/')[-3]
        mesh_id = mesh_res_pkl.split('/')[-1].replace('.pkl','')
        res_save_dir = os.path.join(model_dir,"Attack_Images",mesh_dir,mesh_id)
        experiment_attack(PIXOR, data_loader, device, deform_verts, plot=True, res_save_dir=res_save_dir, images_num=images_num)
    # CUDA_VISIBLE_DEVICES=4 python3 pc_attack.py