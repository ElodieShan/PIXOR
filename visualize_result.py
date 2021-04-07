from load_data import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
##################
# dataset object #
##################


class KittiObject(object):
    """
    Load and parse object data into a usable format.

    """
    def __init__(self, root_dir, split='testing'):
        """
        root_dir contains training and testing folders
        :param root_dir:
        :param split:
        :param args:
        """
        self.root_dir = root_dir
        self.split = split
        # self.split_dir = os.path.join(root_dir, split)
        self.split_dir = os.path.join(root_dir, 'training')

        if split == 'training':
            self.num_samples = 6481
        elif split == 'testing':
            self.num_samples = 1000
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        if self.split == "training":
            file_index = idx
        elif self.split == 'testing':
            file_index = 6481 + idx
        img_filename = os.path.join(self.image_dir, '%06d.png' % file_index)
        return cv2.imread(img_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        if self.split == "training":
            file_index = idx
        elif self.split == 'testing':
            file_index = 6481 + idx
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % file_index)
        return kitti_utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx < self.num_samples)
        if self.split == "training":
            file_index = idx
        elif self.split == 'testing':
            file_index = 6481 + idx
        label_filename = os.path.join(self.label_dir, '%06d.txt' % file_index)
        return kitti_utils.read_label(label_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert (idx < self.num_samples)
        if self.split == "training":
            file_index = idx
        elif self.split == 'testing':
            file_index = 6481 + idx
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (file_index))
        return kitti_utils.load_velo_scan(lidar_filename, dtype, n_vec)


########
# main #
########

if __name__ == '__main__':

    """
    Explore the dataset. For a random selection of indices, the corresponding camera image will be displayed along with 
    all 3D bounding box annotations for the class "Cars". Moreover, the BEV image of the LiDAR point cloud will be 
    displayed with the bounding box annotations and a mask that shows the relevant pixels for the labels used for 
    training the network.
    """

    # root directory of the dataset
    root_dir = 'Data/'

    epoch = 18
    eval_path = os.path.join("Eval", "result_dict_epoch_{}.pkl".format(epoch))

    result_save_dir = os.path.join("Eval", "Figures", str(epoch))
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    with open(eval_path, "rb") as f:
        result_dict_list = pickle.load(f)

    
    # create dataset
    train_dataset = KittiObject(root_dir)

    # select random indices from dataset
    np.random.seed(10)
    ids = np.random.randint(0, 1000, 30)

    # loop over random selection
    for id in ids:
        result_dict = result_dict_list[id]
        # get image, point cloud, labels and calibration
        image = train_dataset.get_image(idx=id)
        labels = train_dataset.get_label_objects(idx=id)
        calib = train_dataset.get_calibration(idx=id)
        point_cloud = train_dataset.get_lidar(idx=id)

        # voxelize the point cloud
        voxel_point_cloud = kitti_utils.voxelize(point_cloud)

        # get BEV image of point cloud
        bev_image = kitti_utils.draw_bev_image(voxel_point_cloud)

        # create empty labels
        regression_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG))
        classification_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA))

        # loop over all annotations for current sample
        for idl, label in enumerate(labels):
            # only display objects labeled as Car
            if label.type == 'Car':
                # compute corners of the bounding box
                bbox_corners_image_coord, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P)
                # print("bbox_corners_camera_coord:", bbox_corners_camera_coord)
                # gt = result_dict["ground_truth_box_corners"]
                # print("ground_truth_box_corners.reshape(4,2):", gt.reshape(gt.shape[0],4,2))
                # print("ground_truth_box_corners.reshape(2,4):", gt.reshape(gt.shape[0],2,4).transpose(0,2,1))
                # pred = result_dict["final_box_predictions"]
                # print("final_box_predictions:",pred[:,2:].reshape(pred.shape[0],2,4).transpose(0,2,1))
                # pred_boxes = pred[:,2:].reshape(pred.shape[0],2,4).transpose(0,2,1)
                # pred_scores = pred[:,1]
                

                # print("final_box_predictions:",pred.reshape(pred.shape[0],2,4).transpose(0,2,1))
                # draw BEV bounding box on BEV image
                bev_image = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord)
                # create labels
                regression_label, classification_label = compute_pixel_labels(regression_label, classification_label,
                                                                   label, bbox_corners_camera_coord)
                # draw 3D bounding box on image
                if bbox_corners_image_coord is not None:
                    image = kitti_utils.draw_projected_box_3d(image, bbox_corners_image_coord)

        pred = result_dict["final_box_predictions"]
        if pred is not None:
            pred_boxes = pred[:,2:].reshape(pred.shape[0],2,4).transpose(0,2,1)
            pred_scores = pred[:,1]
            # print("pred_scores:",pred_scores)
            for i in range(pred_scores.shape[0]):
                pred_score = pred_scores[i]
                pred_box = pred_boxes[i]
                bev_image = kitti_utils.draw_projected_box_bev(bev_image, pred_box, confidence_score=pred_score, color=(0, 0, 255))

        # create binary mask from relevant pixels in label
        label_mask = np.where(np.sum(np.abs(regression_label), axis=2) > 0, 255, 0).astype(np.uint8)

        # remove all points outside the specified area
        idx = np.where(point_cloud[:, 0] > VOX_X_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 0] < VOX_X_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] > VOX_Y_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] < VOX_Y_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] > VOX_Z_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] < VOX_Z_MAX)
        point_cloud = point_cloud[idx]

        # get rectified point cloud for depth information
        point_cloud_rect = calib.project_velo_to_rect(point_cloud[:, :3])

        # color map to indicate depth of point
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        # project point cloud to image plane
        point_cloud_2d = calib.project_velo_to_image(point_cloud[:, :3]).astype(np.int32)

        # draw points
        for i in range(point_cloud_2d.shape[0]):
            depth = point_cloud_rect[i, 2]
            if depth > 0.1:
                color = cmap[int(255 - depth / VOX_X_MAX * 255)-1, :]
                cv2.circle(image, (point_cloud_2d[i, 0], point_cloud_2d[i, 1]), radius=2, color=color, thickness=-1)

        # display images
        # cv2.imshow('Label Mask', c)
        # cv2.imshow('Image', image)
        # cv2.imshow('Image_BEV', bev_image)


        bev_image_save_path = os.path.join(result_save_dir,"{}.png".format(id))
        cv2.imwrite(bev_image_save_path, bev_image)
        # label_mask_save_path = os.path.join("Eval", "KITTI","{}_label_mask.png".format(id))
        # cv2.imwrite(label_mask_save_path, label_mask)

        cv2.waitKey()
