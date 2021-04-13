import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.image import imread
# from det3d.core import box_np_ops
# from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
tsp_axes_limits = [
    [-40, 40], # X axis range
    [0, 60], # Y axis range
    [-2, 10]   # Z axis range
]

velo_axes_limis = [
    [0, 60], # X axis range
    [-40, 40], # Y axis range
    [-2, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

def draw_box(pyplot_axis, vertices, label=None, score=None, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = np.transpose(vertices)[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

    if score is not None:
        label_scores = label + " " + str(score)
    else:
        label_scores = label

    if label_scores is not None:
        if len(axes) > 2:
            pyplot_axis.text(min(vertices[0]), max(vertices[1]),max(vertices[2]), label_scores)
        else:
            pyplot_axis.text(min(vertices[0]), max(vertices[1]), label_scores)



# def get_points_inbox(rbbox_corners, points):
#     surfaces = box_np_ops.corner_to_surfaces_3d(rbbox_corners)
#     indices = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
#     points_inbox_indices = indices.any(axis=1)
#     return points_inbox_indices

# def get_points_inbox_v2(boxes, points):
#     point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
#                     torch.from_numpy(points), torch.from_numpy(boxes)
#                 ).numpy()
#     points_inbox_indices = point_indices.sum(axis=0) == 1
#     return points_inbox_indices

"""
Convenient method for drawing various point cloud projections as a part of frame statistics.
"""
def draw_point_cloud(ax, data, boxes=None, labels=None, scores=None, title=None, gt_boxes=None, gt_labels=None, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None, point_size=0.1, view_points_in_box=True, coordinates="tensorpro"):
    if coordinates == "velodyne":
        axes_limits = velo_axes_limis
        color_axe = 0
    else:
        axes_limits = tsp_axes_limits
        color_axe = 1

    # if view_points_in_box:
    #     points_inbox_indices = get_points_inbox(boxes, data)
    #     ax.scatter(*np.transpose(data[points_inbox_indices][:, axes]), s=point_size+0.05, c='r')
    #     ax.scatter(*np.transpose(data[~points_inbox_indices][:, axes]), s=point_size, c=data[~points_inbox_indices][:, color_axe], cmap='terrain')
    # else:
    # ax.scatter(*np.transpose(data[:, axes]), s=point_size, c=data[:, color_axe], cmap='rainbow')
    ax.scatter(data[:0],data[:1],data[:2], s=point_size, c=data[:, color_axe], cmap='rainbow')

    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)

    if boxes is not None:
        if gt_boxes is not None:
            for i in range(boxes.shape[0]):
                draw_box(ax, boxes[i], axes=axes, color='green')
            for i in range(gt_boxes.shape[0]):
                draw_box(ax, gt_boxes[i], axes=axes, color='red')
        else:
            for i in range(boxes.shape[0]):
                if scores is not None:
                    draw_box(ax, boxes[i], labels[i], scores[i], axes=axes, color='green')
                else:
                    if labels is not None:
                        draw_box(ax, boxes[i], labels[i], axes=axes, color='green')
                    else:
                        draw_box(ax, boxes[i], axes=axes, color='green')

def draw_image(ax, title, pic_path):
    img = imread(pic_path)
    ax.imshow(img)
    ax.set_title(title)



def display_lidar_and_camera(save_path, pic_path, data, boxes=None, points_size=0.2, view=False):
    # points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, data.shape[0], points_step)
    
    print(points_step)
    print(point_size)
    print(velo_range)

    if view:
        # Draw point cloud data as plane projections
        f, ax3 = plt.subplots(2, 1, figsize=(15, 25))
        draw_image(
            ax3[0], 
            'Camera View of Scene', 
            pic_path
        )
        draw_point_cloud(
            ax3[1], 
            data,
            boxes,
            'Tensorpro Point Cloud, XY projection (Z = 0)', 
            axes=[0, 1] # X and Y axes
        )
        plt.savefig(save_path)
        plt.show()

def display_2lidar_and_camera(save_path, pic_path, data_tsp, data_velo, boxes_tsp=None, boxes_velo=None, labels_tsp=None, labels_velo=None, scores_tsp=None, scores_velo=None, points_size=5, view_3d=False, coordinates="tensorpro"):
    # points_step = int(1. / points)
    # point_size = 0.01 * (1. / points)
    # velo_range = range(0, data.shape[0], points_step)
    
    # print(points_step)
    # print(point_size)
    # print(velo_range)
    #　滤除tensorpro中y值大于150的点
    data_tsp = data_tsp[np.where(data_tsp[:,1]<150)]
    if not view_3d:
        # Draw point cloud data as plane projections
        f, ax3 = plt.subplots(3, 1, figsize=(15, 20))
        plt.tight_layout()
        draw_image(
            ax3[0],
            'Camera View of Scene', 
            pic_path
        )
        draw_point_cloud(
            ax3[1], 
            data_tsp,
            boxes_tsp,
            labels_tsp,
            scores_tsp,
            'Tensorpro Point Cloud, XY projection (Z = 0)', 
            axes=[0, 1], # X and Y axes
            point_size = points_size,
            coordinates = coordinates
        )
        draw_point_cloud(
            ax3[2],
            data_velo,
            boxes_velo,
            labels_velo,
            scores_velo,
            title = 'Velodyne Point Cloud, XY projection (Z = 0)', 
            axes=[0, 1], # X and Y axes
            point_size= points_size,
            coordinates = coordinates
        )
        plt.savefig(save_path)
        # plt.show()

    if view_3d:
        f2 = plt.figure(figsize=(20, 30))
        ax1 = f2.add_subplot(311)
        draw_image(
            ax1, 
            'Camera View of Scene', 
            pic_path
        )

        ax2 = f2.add_subplot(323, projection='3d')
        if coordinates == "velodyne":
            ax2.view_init(45,-150)
        else:
            ax2.view_init(45,-60)
        # ax2.grid(False)
        draw_point_cloud(
            ax2, data_tsp, boxes_tsp, labels_tsp, scores_tsp,
            title = 'Tensorpro 3D Point Cloud ',point_size = points_size, coordinates = coordinates)

        ax3 = f2.add_subplot(324)
        draw_point_cloud(
            ax3, data_tsp, boxes_tsp, labels_tsp, scores_tsp,
            title = 'Tensorpro Point Cloud, XY projection (Z = 0)',
            axes=[0, 1],
            point_size = points_size,
            coordinates = coordinates
        )

        ax4 = f2.add_subplot(325, projection='3d')
        if coordinates == "velodyne":
            ax4.view_init(45,-150)
        else:
            ax4.view_init(45,-60)
        # ax4.grid(False)
        draw_point_cloud(
            ax4, data_velo, boxes_velo, labels_velo, scores_velo,
            title = 'Velodyne 3D Point Cloud ',point_size = points_size, coordinates = coordinates)

        ax5 = f2.add_subplot(326)
        draw_point_cloud(
            ax5, data_velo, boxes_velo, labels_velo, scores_velo,
            title = 'Velodyne Point Cloud, XY projection (Z = 0)',
            axes=[0, 1],
            point_size = points_size,
            coordinates = coordinates
        )
        plt.savefig(save_path)
        plt.close('all')
        # plt.show()

def display_pred_and_gt(save_path, data, boxes_dt=None, boxes_gt=None, labels_dt=None, labels_gt=None, scores_dt=None, points_size=5, view_3d=False, coordinates="tensorpro"):
    data = data[np.where(data[:,1]>0)]

    if view_3d:
        f = plt.figure(figsize=(20, 30))

        ax1 = f.add_subplot(321, projection='3d')
        if coordinates == "velodyne":
            ax1.view_init(45,-150)
        else:
            ax1.view_init(45,-60)
        draw_point_cloud(
            ax1, data, boxes_dt, labels_dt, scores_dt,
            title = '3D Point Cloud & Detection Result',
            point_size = points_size, coordinates = coordinates)
        
        ax2 = f.add_subplot(322)
        draw_point_cloud(
            ax2, data, boxes_dt, labels_dt, scores_dt,
            title = 'Point Cloud & Detection Result, XY projection (Z = 0)',
            axes=[0, 1],
            point_size = points_size,
            coordinates = coordinates
        )

        ax3 = f.add_subplot(323, projection='3d')
        if coordinates == "velodyne":
            ax3.view_init(45,-150)
        else:
            ax3.view_init(45,-60)
        draw_point_cloud(
            ax3, data, boxes_gt, labels_gt,
            title='3D Point Cloud & Grount Truth Result',
            point_size = points_size, coordinates = coordinates)
        
        ax4 = f.add_subplot(324)
        draw_point_cloud(
            ax4, data, boxes_gt, labels_gt,
            title = 'Point Cloud & Grount Truth  Result, XY projection (Z = 0)',
            axes=[0, 1],
            point_size = points_size,
            coordinates = coordinates
        )


        ax5 = f.add_subplot(325, projection='3d')
        if coordinates == "velodyne":
            ax1.view_init(45,-150)
        else:
            ax1.view_init(45,-60)
        draw_point_cloud(
            ax5, data, boxes_dt, labels_dt, scores_dt,
            title = '3D Point Cloud & Result Compare',
            gt_boxes=boxes_gt, gt_labels=labels_gt, 
            point_size = points_size, coordinates = coordinates,
            view_points_in_box=False)
        
        ax6 = f.add_subplot(326)
        draw_point_cloud(
            ax6, data, boxes_dt, labels_dt, scores_dt,
            title = 'Point Cloud & Result Compare, XY projection (Z = 0)',
            axes=[0, 1],
            gt_boxes=boxes_gt, gt_labels=labels_gt, 
            point_size = points_size,
            coordinates = coordinates,
            view_points_in_box=False
        )
        plt.savefig(save_path)
        plt.close('all')