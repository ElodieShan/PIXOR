
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

from loss import CustomLoss
from datagen import get_data_loader
from model import PIXOR
from utils import get_model_name, load_config, get_logger, plot_bev, plot_label_map, plot_pr_curve, get_bev
from postprocess import filter_pred, compute_matches, compute_ap
from main import *

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


def attack_model(model, device, exp_name='default'):
    root = "/home/elodie"
    suffix = 'try_raw'

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter(logdir=os.path.join(root, 'data/runs', TIMESTAMP + '_' + suffix))

    # start optimization loop
    mesh0 = p3d.utils.ico_sphere(2, device)
    print("mesh0:",mesh0)
    physical_transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))

    deform_verts = mesh0.verts_packed().new_zeros(mesh0.verts_packed().shape).requires_grad_()
    optimizer = torch.optim.Adam([deform_verts], lr=0.001)

    # Niter = 2
    Niter = 37120
    plot_period = 1000

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
        # data_dict = copy.deepcopy(data_dict_show)
        # load_data_to_gpu(data_dict)

        #--------- Load Data-------
        image_id = i%len(model.train_loader.dataset)
        label_map, label_list, gt_boxes_lidar = model.train_loader.dataset.get_label(image_id, return_gt_boxes=True)
        if gt_boxes_lidar is None:
            # print("None of car boxes! Skip..")
            continue
        input, _, _ = model.train_loader.dataset[image_id]
        # if torch.isnan(input).all():
        #     print("Find Nan")
        # if torch.isinf(input).all():
        #     print("Find Inf")

        # print("input:",input)
        # print("input:",input[input>0])

        model.train_loader.dataset.reg_target_transform(label_map)
        points = model.train_loader.dataset.get_points(image_id)
        data_dict = {
            'input': input,
            'points': torch.from_numpy(points),
            'label_map': label_map,
            'image_id': image_id,
            'gt_boxes_corner': np.array(label_list),
            'gt_boxes': torch.from_numpy(gt_boxes_lidar).to(device),
        }
        # print("data_dict['gt_boxes_corner']:",data_dict['gt_boxes_corner'])
        # model(data_dict)

        # for bb in data_dict['gt_boxes'][data_dict['gt_boxes'][..., 7] == 1][..., :7]:
        new_pts_num = 0
        for bb in data_dict['gt_boxes']:
            mesh = mesh0.offset_verts(torch.tanh(deform_verts) * bound)
            adv_bb = adv_bbox(bb)
            # print("bb:",bb)
            new_pts = sample_pc(mesh, bb, adv_bb, device=device)
            # print("new_pts:",new_pts)

            if new_pts is not None:
                new_pts[:,0] = (new_pts[:,0] - x_MIN)/x_DIVISION
                new_pts[:,1] = (new_pts[:,1] - y_MIN)/y_DIVISION
                new_pts[:,2] = (new_pts[:,2] - z_MIN)/z_DIVISION
                new_pts_index = new_pts.long()
                
                index_mask = torch.vstack((new_pts_index[:,0]<x_INDEX_MAX, new_pts_index[:,1]<y_INDEX_MAX, new_pts_index[:,2]<z_INDEX_MAX)).all(dim=0)
                if index_mask.sum() > 0:
                    new_pts_index = new_pts_index[index_mask][:, [2, 1,0]]
                    new_pts_index_num = new_pts_index.shape[0]
                    new_pts_index=new_pts_index.permute(1,0)
                    data_dict['input'].index_put_(tuple(new_pts_index), torch.ones(new_pts_index_num))
                new_pts_num += index_mask.sum()
            else:
                warnings.warn("Warning Lidar shoot nothing")
        if new_pts_num ==0:
            warnings.warn("Mesh remains None!")
            # print("Mesh remains None! Skip..")
            continue

        # data_dict['input'] = model.train_loader.dataset.get_top_view_maps(data_dict['points'].detach().numpy())
        pred_dicts, _ = model(data_dict, device=device, if_raw=True)
        # pred_scores = pred_dicts['final_scores']
        # print("pred_scores:",pred_scores.shape)
        pred_scores = pred_dicts['cls_pred'][pred_dicts['cls_targets'] == 1]

        # print("pred_scores:",pred_scores.shape)
        # corners, scores = filter_pred(config, pred)
        #     sample_trg = p3d.ops.sample_points_from_meshes(trg_mesh, 5000)
        #     sample_src = p3d.ops.sample_points_from_meshes(new_src_mesh, 5000)

        #     loss_chamfer, _ = p3d.loss.chamfer_distance(sample_trg, sample_src)

        pred_scores=torch.clamp(pred_scores, 0, 0.9999)

        loss_adv = - (1 - pred_scores).log().mean()
        #     loss_edge = p3d.loss.mesh_edge_loss(mesh)
        #     loss_normal = p3d.loss.mesh_normal_consistency(mesh)
        # loss = loss_adv
        loss_laplacian = p3d.loss.mesh_laplacian_smoothing(mesh, method="uniform")
        loss = loss_adv + loss_laplacian * w_laplacian
        #     loss = loss_adv * w_adv + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

        #     adv_losses.append(loss_adv)
        #     edge_losses.append(loss_edge)
        #     normal_losses.append(loss_normal)
        #     laplacian_losses.append(loss_laplacian)
        #     losses.append(loss)

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
            save_path_dir = os.path.join('Figures',exp_name, TIMESTAMP, 'mesh')
            if not os.path.exists(save_path_dir):
                os.makedirs(save_path_dir)
            save_path = os.path.join('Figures',exp_name, TIMESTAMP, 'mesh', ('%s_mesh.png' % i))
            plt.savefig(save_path)
            fig = ax.get_figure()
            writer.add_figure('mesh', fig, i)

            res_save_dir = os.path.join('logs','attack',exp_name,TIMESTAMP)
            if not os.path.exists(res_save_dir):
                os.makedirs(res_save_dir)
            res_save_path = os.path.join('logs','attack',exp_name, TIMESTAMP,('verts_%s.pkl' % i))
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

    print("exp_name:",exp_name)
    # res_save_path = os.path.join('logs',exp_name, ('verts_%s.pkl' % TIMESTAMP))
    res_save_path = os.path.join('logs','attack',exp_name, TIMESTAMP,('verts_%s.pkl' % TIMESTAMP))

    with open(res_save_path, 'wb') as f:
        pickle.dump(deform_verts.detach(), f)


def eval_batch_attack(config, net, deform_verts, loss_fn, loader, device, eval_range='all'):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)
    
    # load attack mesh 
    mesh0 = p3d.utils.ico_sphere(2, device)
    physical_transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))

    attack_mesh = mesh0.offset_verts(deform_verts).verts_packed()

    cls_loss = 0
    loc_loss = 0
    all_scores = []
    all_matches = []
    log_images = []
    gts = 0
    preds = 0
    t_fwd = 0
    t_nms = 0

    log_img_list = random.sample(range(len(loader.dataset)), 10)

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            tic = time.time()
            input, label_map, image_id = data
            input = input.to(device)
            label_map = label_map.to(device)
            tac = time.time()
            predictions = net(input)
            t_fwd += time.time() - tac
            loss, cls, loc = loss_fn(predictions, label_map)
            cls_loss += cls
            loc_loss += loc 
            t_fwd += (time.time() - tic)
            
            toc = time.time()
            # Parallel post-processing
            predictions = list(torch.split(predictions.cpu(), 1, dim=0))
            batch_size = len(predictions)
            with Pool (processes=3) as pool:
                preds_filtered = pool.starmap(filter_pred, [(config, pred) for pred in predictions])
            t_nms += (time.time() - toc)
            args = []
            for j in range(batch_size):
                _, label_list = loader.dataset.get_label(image_id[j].item())
                corners, scores = preds_filtered[j]
                gts += len(label_list)
                preds += len(scores)
                all_scores.extend(list(scores))
                if image_id[j] in log_img_list:
                    input_np = input[j].cpu().permute(1, 2, 0).numpy()
                    pred_image = get_bev(input_np, corners)
                    log_images.append(pred_image)

                arg = (np.array(label_list), corners, scores)
                args.append(arg)

            # Parallel compute matchesi
            
            with Pool (processes=3) as pool:
                matches = pool.starmap(compute_matches, args)
            
            for j in range(batch_size):
                all_matches.extend(list(matches[j][1]))
            
            #print(time.time() -tic)
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches = all_matches[sort_ids[::-1]]

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['Forward Pass Time'] = t_fwd/len(loader.dataset)
    metrics['Postprocess Time'] = t_nms/len(loader.dataset) 

    cls_loss = cls_loss / len(loader)
    loc_loss = loc_loss / len(loader)
    metrics['loss'] = cls_loss + loc_loss

    return metrics, precisions, recalls, log_images

def eval_one_attack(deform_verts, net, loss_fn, config, loader, image_id, device, plot=False, verbose=False, name=None):
    input, label_map, image_id = loader.dataset[image_id]
    input = input.to(device)
    label_map, label_list, gt_boxes3d_lidar = loader.dataset.get_label(image_id, return_gt_boxes=True)

    loader.dataset.reg_target_transform(label_map)
    label_map = torch.from_numpy(label_map).permute(2, 0, 1).unsqueeze_(0).to(device)
    # Forward Pass
    t_start = time.time()
    if gt_boxes3d_lidar is not None:
        gt_boxes3d_lidar = torch.from_numpy(gt_boxes3d_lidar).to(device)
        new_pts_num = 0
        for bb in gt_boxes3d_lidar:
            # ----------- Load attack mesh 
            bound = deform_verts.new([0.1, 0.1, 0])
            mesh0 = p3d.utils.ico_sphere(2, device)
            physical_transforms(mesh0, scale=torch.cuda.FloatTensor([0.7, 0.7, 0.5]))

            attack_mesh = mesh0.offset_verts(torch.tanh(deform_verts) * bound)
            adv_bb = adv_bbox(bb)
            # print("bb:",bb)
            # print("attack_mesh:",attack_mesh)
            new_pts = sample_pc(attack_mesh, bb, adv_bb, device=device)

            if new_pts is not None:
                new_pts[:,0] = (new_pts[:,0] - x_MIN)/x_DIVISION
                new_pts[:,1] = (new_pts[:,1] - y_MIN)/y_DIVISION
                new_pts[:,2] = (new_pts[:,2] - z_MIN)/z_DIVISION
                new_pts_index = new_pts.long()
                
                index_mask = torch.vstack((new_pts_index[:,0]<x_INDEX_MAX, new_pts_index[:,1]<y_INDEX_MAX, new_pts_index[:,2]<z_INDEX_MAX)).all(dim=0)
                if index_mask.sum() > 0:
                    new_pts_index = new_pts_index[index_mask][:, [2, 1,0]]
                    new_pts_index_num = new_pts_index.shape[0]
                    new_pts_index=new_pts_index.permute(1,0)
                    input.index_put_(tuple(new_pts_index), torch.ones(new_pts_index_num).to(device))
                new_pts_num += index_mask.sum()
            else:
                warnings.warn("Warning Lidar shoot nothing")
        if new_pts_num ==0:
            warnings.warn("Mesh remains None!")
            # print("Mesh remains None! Skip..")
            # continue

    pred = net(input.unsqueeze(0))
    t_forward = time.time() - t_start

    loss, cls_loss, loc_loss = loss_fn(pred, label_map)
    pred.squeeze_(0)
    cls_pred = pred[0, ...]

    if verbose:
        print("Forward pass time", t_forward)


    # Filter Predictions
    t_start = time.time()
    corners, scores = filter_pred(config, pred)
    t_post = time.time() - t_start

    if verbose:
        print("Non max suppression time:", t_post)

    gt_boxes = np.array(label_list)
    gt_match, pred_match, overlaps = compute_matches(gt_boxes,
                                        corners, scores, iou_threshold=0.7)

    num_gt = len(label_list)
    num_pred = len(scores)
    input_np = input.cpu().permute(1, 2, 0).numpy()
    pred_image = get_bev(input_np, corners)

    if plot == True:
        # Visualization
        folder = os.path.join("Figures", name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plot_bev(input_np, label_list, save_path=os.path.join(folder, '{}_GT.png'.format(image_id)))
        plot_bev(input_np, corners, save_path=os.path.join(folder, '{}_Prediction.png'.format(image_id)))

        # plot_bev(input_np, label_list, window_name='GT')
        # plot_bev(input_np, corners, window_name='Prediction')
        # plot_label_map(cls_pred.numpy())

    return num_gt, num_pred, scores, pred_image, pred_match, loss.item(), t_forward, t_post

def eval_attack_set(deform_verts, config, net, loss_fn, loader, device, e_range='all', result_save_pth=None):
    net.eval()
    if config['mGPUs']:
        net.module.set_decode(True)
    else:
        net.set_decode(True)


    # ----------- Load attack mesh end

    t_fwds = 0
    t_post = 0
    loss_sum = 0

    img_list = range(len(loader.dataset))
    if e_range != 'all':
        e_range = min(e_range, len(loader.dataset))
        img_list = random.sample(img_list, e_range)

    log_img_list = random.sample(img_list, 10)

    gts = 0
    preds = 0
    all_scores = []
    all_matches = []
    log_images = []

    with torch.no_grad():
        for image_id in tqdm(img_list):
            #tic = time.time()
            num_gt, num_pred, scores, pred_image, pred_match, loss, t_forward, t_nms = \
                eval_one_attack(deform_verts, net, loss_fn, config, loader, image_id, device, plot=False)
            gts += num_gt
            preds += num_pred
            loss_sum += loss
            all_scores.extend(list(scores))
            all_matches.extend(list(pred_match))

            t_fwds += t_forward
            t_post += t_nms

            if image_id in log_img_list:
                log_images.append(pred_image)
            #print(time.time() - tic)
            
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    sort_ids = np.argsort(all_scores)
    all_matches_sorted = all_matches[sort_ids[::-1]]
    if result_save_pth is not None:
        res_dict = {
            "all_scores":all_scores,
            "all_matches":all_matches,
            "sort_ids":sort_ids,
            "all_matches_sorted":all_matches_sorted
        }
        with open(result_save_pth, "wb") as f:
            pickle.dump(res_dict, f)

    metrics = {}
    AP, precisions, recalls, precision, recall = compute_ap(all_matches_sorted, gts, preds)
    metrics['AP'] = AP
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['loss'] = loss_sum / len(img_list)
    metrics['Forward Pass Time'] = t_fwds / len(img_list)
    metrics['Postprocess Time'] = t_post / len(img_list)

    return metrics, precisions, recalls, log_images

def experiment_attack(exp_name, device, mesh_res_pkl=None, eval_range='all', plot=True):
    config, _, _, _ = load_config(exp_name)
    net, loss_fn = build_model(config, device, train=False)
    state_dict = torch.load(get_model_name(config), map_location=device)
    if config['mGPUs']:
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    train_loader, val_loader = get_data_loader(config['batch_size'], config['use_npy'], geometry=config['geometry'],
                                               frame_range=config['frame_range'])
    
    print("mesh_res_pkl:",mesh_res_pkl)
    with open(mesh_res_pkl, 'rb') as f:
        mesh_res = pickle.load(f)
    print("mesh_res:",mesh_res)
    # Train Set
    # result_save_pth = mesh_res_pkl.replace('.pkl','_trainset-IOU7.pkl')
    # train_metrics, train_precisions, train_recalls, _ = eval_attack_set(mesh_res, config, net, loss_fn, train_loader, device, eval_range, result_save_pth=result_save_pth)
    # print("Training mAP", train_metrics['AP'])
    # fig_name = "PRCurve_train_IOU7-attack" + config['name']
    # legend = "AP={:.1%} @IOU=0.7-attack".format(train_metrics['AP'])

    # # legend = "AP={:.1%} @IOU=0.5".format(train_metrics['AP'])
    # plot_pr_curve(train_precisions, train_recalls, legend, name=fig_name)

    result_save_pth = mesh_res_pkl.replace('.pkl','_valset-IOU7.pkl')

    # Val Set
    val_metrics, val_precisions, val_recalls, _ = eval_attack_set(mesh_res,config,  net, loss_fn, val_loader, device, eval_range, result_save_pth=result_save_pth)

    print("Validation mAP", val_metrics['AP'])
    print("Net Fwd Pass Time on average {:.4f}s".format(val_metrics['Forward Pass Time']))
    print("Nms Time on average {:.4f}s".format(val_metrics['Postprocess Time']))

    fig_name = "PRCurve_val_IOU7-attack" + config['name']
    legend = "AP={:.1%} @IOU=0.7-attack".format(val_metrics['AP'])

    # legend = "AP={:.1%} @IOU=0.5".format(val_metrics['AP'])
    plot_pr_curve(val_precisions, val_recalls, legend, name=fig_name)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    device = 'cuda:0'

    print("Using device", device)
    # ---------- Load PIXOR Detector ------------
    exp_name = 'test3' 
    # mesh_res_pkl = "logs/attack/test3/2021-03-25T14-11-14/verts_9000.pkl"
    # experiment_attack(exp_name, device, mesh_res_pkl=mesh_res_pkl, eval_range='all', plot=True)
    
    det = LoadPIXOR(exp_name=exp_name, device=device)
    attack_model(det, device, exp_name=exp_name)