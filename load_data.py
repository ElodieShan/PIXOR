from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import cv2
import kitti_utils
from config import *
import time
import ctypes

############################
# custom collate functions #
############################


def my_collate_test(batch):
    """
    Collate function for test dataset. How to concatenate individual samples to a batch.
    Point Clouds will be stacked along first dimension, labels and calibration objects will be returned as a list
    :param batch: list containing a tuple of items for each sample
    :return: batch data in desired form
    """

    point_clouds = []
    labels = []
    calibs = []
    training_labels = []
    tuple_shape = 3
    for tuple_id, tuple in enumerate(batch):
        point_clouds.append(tuple[0])
        labels.append(tuple[1])
        calibs.append(tuple[2])
        if len(tuple)>3:
            tuple_shape = len(tuple)
            training_labels.append(tuple[3])

    point_clouds = torch.stack(point_clouds)
    if tuple_shape>3:
        return point_clouds, labels, calibs, training_labels
    return point_clouds, labels, calibs


def my_collate_train(batch):
    """
    Collate function for training dataset. How to concatenate individual samples to a batch.
    Point Clouds and labels will be stacked along first dimension
    :param batch: list containing a tuple of items for each sample
    :return: batch data in desired form
    """

    point_clouds = []
    labels = []
    for tuple_id, tuple in enumerate(batch):
        point_clouds.append(tuple[0])
        labels.append(tuple[1])

    point_clouds = torch.stack(point_clouds)
    labels = torch.stack(labels)
    return point_clouds, labels


########################
# compute pixel labels #
########################

def compute_pixel_labels(regression_label, classification_label, label, bbox_corners_camera_coord):
    """
    Compute the label that will be fed into the network from the bounding box annotations of the respective point cloud.
    :param: regression_label: emtpy numpy array | shape: [OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG]
    :param: classification_label: emtpy numpy array | shape: [OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA]
    :param label: 3D label object containing bounding box information
    :param bbox_corners_camera_coord: corners of the bounding box | shape: [8, 3]
    :return: regression_label and classification_label filled with relevant label information
    """

    # get label information
    angle_rad = label.ry  # rotation of bounding box
    center_x_m = label.t[0]
    center_y_m = label.t[2]
    length_m = label.length
    width_m = label.width

    # extract corners of BEV bounding box
    bbox_corners_x = bbox_corners_camera_coord[:4, 0]
    bbox_corners_y = bbox_corners_camera_coord[:4, 2]

    # convert coordinates from m to pixels
    corners_x_px = ((bbox_corners_x - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
    corners_y_px = (INPUT_DIM_0 - ((bbox_corners_y - VOX_X_MIN) // VOX_X_DIVISION)).astype(np.int32)
    bbox_corners = np.vstack((corners_x_px, corners_y_px)).T

    # create a pixel mask of the target bounding box
    canvas = np.zeros((INPUT_DIM_0, INPUT_DIM_1, 3))
    canvas = cv2.fillPoly(canvas, pts=[bbox_corners], color=(255, 255, 255))

    # resize label to fit output shape
    canvas_resized = cv2.resize(canvas, (OUTPUT_DIM_1, OUTPUT_DIM_0), interpolation=cv2.INTER_NEAREST)
    bbox_mask = np.where(np.sum(canvas_resized, axis=2) == 765, 1, 0).astype(np.uint8)[:, :, np.newaxis]

    # get location of each pixel in m
    x_lin = np.linspace(VOX_Y_MIN, VOX_Y_MAX-0.4, OUTPUT_DIM_1)
    y_lin = np.linspace(VOX_X_MAX, VOX_X_MIN+0.4, OUTPUT_DIM_0)
    px_x, px_y = np.meshgrid(x_lin, y_lin)

    # create regression target
    target = np.array([[np.cos(angle_rad), np.sin(angle_rad), -center_x_m, -center_y_m, np.log(width_m), np.log(length_m)]])
    target = np.tile(target, (OUTPUT_DIM_0, OUTPUT_DIM_1, 1))

    # take offset from pixel as regression target for bounding box location
    target[:, :, 2] += px_x
    target[:, :, 3] += px_y

    # normalize target
    target = (target - REG_MEAN) / REG_STD

    # zero-out non-relevant pixels
    target *= bbox_mask

    # add current target to label for currently inspected point cloud
    regression_label += target
    classification_label += bbox_mask

    return regression_label, classification_label


###################
# dataset classes #
###################


class PointCloudDataset(Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, root_dir, split='training', device=torch.device('cpu'), show_times=True, get_image=False, \
                        use_voxelize_density=False, use_ImageSets=False, get_training_label_when_test=False):
        """
        Dataset for training and testing containing point cloud, calibration object and in case of training labels
        :param root_dir: root directory of the dataset
        :param split: training or testing split of the dataset
        :param device: device on which dataset will be used
        :param show_times: show times of each step of the data loading (debug)
        """

        self.show_times = show_times  # debug
        self.get_image = get_image  # load camera image

        self.use_ImageSets = use_ImageSets
        self.device = device
        self.root_dir = root_dir
        self.split = split
        # self.split_dir = os.path.join(root_dir, split)
        self.split_dir = os.path.join(root_dir, 'training')
        # self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        self.get_training_label_when_test = get_training_label_when_test
        print("get_training_label_when_test:",get_training_label_when_test)

        if self.use_ImageSets:
            if split == 'training':
                self.num_samples = 3712
                self.trainset_list = self.load_imageset(is_train=True)
            elif split == 'testing' or split == 'val':
                self.num_samples = 3769
                self.val_list = self.load_imageset(is_train=False)
            else:
                print('Unknown split: %s' % (split))
                exit(-1)
        else:
            if split == 'training':
                self.num_samples = 6481
            elif split == 'testing':
                self.num_samples = 1000
            else:
                print('Unknown split: %s' % (split))
                exit(-1)

        self.use_voxelize_density = use_voxelize_density
        # paths to camera, lidar, calibration and label directories
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.image_dir = os.path.join(self.split_dir, 'image_2')

    def __len__(self):
        # Denotes the total number of samples
        return self.num_samples

    def get_file_index(self, idx):
        if self.use_ImageSets:
            if self.split == "training":
                file_index = self.trainset_list[idx]
            elif self.split == 'testing' or self.split == 'val':
                file_index = self.val_list[idx]
        else:
            if self.split == "training":
                file_index = idx
            elif self.split == 'testing':
                file_index = 6481 + idx
        return file_index

    def get_top_view_maps(self, points):
        scan = np.zeros((INPUT_DIM_0,INPUT_DIM_1,INPUT_DIM_2), dtype=np.float32)
        c_data = ctypes.c_void_p(scan.ctypes.data)
        c_points = ctypes.c_void_p(points.ctypes.data)
        num = points.shape[0]
        ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so').createTopViewMapsfromPoints(c_data, c_points, num)
        return scan

    def get_top_view_density_maps(self, points):
        scan = np.zeros((INPUT_DIM_0,INPUT_DIM_1,INPUT_DIM_2), dtype=np.float32)
        c_data = ctypes.c_void_p(scan.ctypes.data)
        c_points = ctypes.c_void_p(points.ctypes.data)
        num = points.shape[0]
        ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so').createTopViewDensityMapsfromPoints(c_data, c_points, num)
        return scan
        
    def get_top_view_maps_from_file(self, filename):
        c_name = bytes(filename, 'utf-8')
        scan = np.zeros((INPUT_DIM_0,INPUT_DIM_1,INPUT_DIM_2), dtype=np.float32)
        # print(scan.shape)
        c_data = ctypes.c_void_p(scan.ctypes.data)
        ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so').createTopViewMaps(c_data, c_name)
        return scan

    def __getitem__(self, index):
        
        # start time
        get_item_start_time = time.time()

        # elodie
        file_index = self.get_file_index(index)

        # get point cloud
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % file_index)
        lidar_data = kitti_utils.load_velo_scan(lidar_filename)

        # time for loading point cloud
        read_point_cloud_end_time = time.time()
        read_point_cloud_time = read_point_cloud_end_time - get_item_start_time

        # voxelize point cloud
        # voxel_point_cloud = torch.tensor(kitti_utils.voxelize(point_cloud=lidar_data), requires_grad=True, device=self.device).float()
        if self.use_voxelize_density:
            # voxel_point_cloud = torch.from_numpy(kitti_utils.voxelize_density(point_cloud = lidar_data)).to(self.device)
            voxel_point_cloud = torch.from_numpy(self.get_top_view_density_maps(lidar_data)).to(self.device)
        else:
            # voxel_point_cloud = torch.from_numpy(self.get_top_view_maps_from_file(lidar_filename)).to(self.device)
            # voxel_point_cloud = torch.from_numpy(kitti_utils.voxelize(point_cloud = lidar_data)).to(self.device)
            voxel_point_cloud = torch.from_numpy(self.get_top_view_maps(lidar_data)).to(self.device)
        # time for voxelization
        voxelization_end_time = time.time()
        voxelization_time = voxelization_end_time - read_point_cloud_end_time

        # channels along first dimensions according to PyTorch convention
        voxel_point_cloud = voxel_point_cloud.permute([2, 0, 1])

        # get image
        if self.get_image:
            image_filename = os.path.join(self.image_dir, '%06d.png' % file_index)
            image = kitti_utils.get_image(image_filename)

        # get current time
        read_labels_start_time = time.time()

        # get calibration
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % file_index)
        calib = kitti_utils.Calibration(calib_filename)

        # get labels
        label_filename = os.path.join(self.label_dir, '%06d.txt' % file_index)
        labels = kitti_utils.read_label(label_filename)

        read_labels_end_time = time.time()
        read_labels_time = read_labels_end_time - read_labels_start_time

        # compute network label
        if self.split == 'training' or self.split == 'val':
            # get current time
            compute_label_start_time = time.time()

            # create empty pixel labels
            regression_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG))
            classification_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA))

            # iterate over all 3D label objects in list
            for label in labels:
                if label.type == 'Car':
                    # compute corners of 3D bounding box in camera coordinates
                    _, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P, scale=1.0)
                    # get pixel label for classification and BEV bounding box
                    regression_label, classification_label = compute_pixel_labels\
                        (regression_label, classification_label, label, bbox_corners_camera_coord)

            # stack classification and regression label
            regression_label = torch.tensor(regression_label, device=self.device).float()
            classification_label = torch.tensor(classification_label, device=self.device).float()
            training_label = torch.cat((regression_label, classification_label), dim=2)

            # get time for computing pixel label
            compute_label_end_time = time.time()
            compute_label_time = compute_label_end_time - compute_label_start_time

            # total time for data loading
            get_item_end_time = time.time()
            get_item_time = get_item_end_time - get_item_start_time

            if self.show_times:
                print('---------------------------')
                print('Get Item Time: {:.4f} s'.format(get_item_time))
                print('---------------------------')
                print('Read Point Cloud Time: {:.4f} s'.format(read_point_cloud_time))
                print('Voxelization Time: {:.4f} s'.format(voxelization_time))
                print('Read Labels Time: {:.4f} s'.format(read_labels_time))
                print('Compute Labels Time: {:.4f} s'.format(compute_label_time))
            return voxel_point_cloud, training_label
    

        else:
            if self.get_image:
                return image, voxel_point_cloud, labels, calib
            else:
                if self.get_training_label_when_test:
                    print("OK!")
                    # create empty pixel labels
                    regression_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG))
                    classification_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA))
                    # iterate over all 3D label objects in list
                    for label in labels:
                        if label.type == 'Car':
                            # compute corners of 3D bounding box in camera coordinates
                            _, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P, scale=1.0)
                            # get pixel label for classification and BEV bounding box
                            regression_label, classification_label = compute_pixel_labels\
                                (regression_label, classification_label, label, bbox_corners_camera_coord)
                    # stack classification and regression label
                    regression_label = torch.tensor(regression_label, device=self.device).float()
                    classification_label = torch.tensor(classification_label, device=self.device).float()
                    training_label = torch.cat((regression_label, classification_label), dim=2)
                    return voxel_point_cloud, labels, calib, training_label
                else:
                    return voxel_point_cloud, labels, calib

    def load_imageset(self, is_train=True):
            path = "ImageSets"
            if is_train:
                path = os.path.join(path, "train.txt")
            else:
                path = os.path.join(path, "val.txt")

            with open(path, 'r') as f:
                lines = f.readlines() # get rid of \n symbol
                names = []
                for line in lines[:-1]:
                    if int(line[:-1]) < 7482:
                        names.append(int(line[:-1]))

                # Last line does not have a \n symbol
                names.append(int(lines[-1][:6]))
                # print(names[-1])
                print("There are {} images in txt file".format(len(names)))

                return names

#################
# load datasets #
#################

def load_dataset(root='Data/', batch_size=1, train_val_split=0.9, test_set=False,
                 device=torch.device('cpu'), show_times=False, \
                 use_voxelize_density=False, use_ImageSets=False, get_training_label_when_test=False):
    """
    Create a data loader that reads in the data from a directory of png-images
    :param device: device of the model
    :param root: root directory of the image data
    :param batch_size: batch-size for the data loader
    :param train_val_split: fraction of the available data used for training
    :param test_set: if True, data loader will be generated that contains only a test set
    :param show_times: display times for each step of the data loading
    :return: torch data loader object
    """

    # speed up data loading on gpu
    if device != torch.device('cpu'):
        num_workers = 4
    else:
        num_workers = 0
    print("num_workers:",num_workers)
    # create training and validation set
    if not test_set:
        # create customized dataset class
        dataset = PointCloudDataset(root_dir=root, device=device, split='training', show_times=show_times, \
                            use_voxelize_density=use_voxelize_density, use_ImageSets=use_ImageSets)

        # number of images used for training and validation
        n_images = dataset.__len__()
        n_train = int(train_val_split * n_images)
        n_val = n_images - n_train

        # generated training and validation set
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # create data_loaders
        data_loader = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_train,
                                num_workers=num_workers),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_train,
                              num_workers=num_workers)
        }
    # create test set
    else:

        test_dataset = PointCloudDataset(root_dir=root, device=device, split='testing', \
        use_voxelize_density=use_voxelize_density, use_ImageSets=use_ImageSets, get_training_label_when_test=get_training_label_when_test)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_test,
                                 num_workers=num_workers, drop_last=True)

    return data_loader


if __name__ == '__main__':

    # create data loader
    root_dir = 'Data/'
    batch_size = 1
    device = torch.device('cpu')

    # data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device, show_times=True)
    # for batch_id, (batch_data, batch_labels) in enumerate(data_loader['val']):
    #     print("batch_id:",batch_id,"\nbatch_data:",batch_data[batch_data>0])

    # Test
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device, test_set=True)

    for batch_id, (batch_data, batch_labels, batch_calib) in enumerate(data_loader):
        print("batch_id:",batch_id,"\nbatch_data:",batch_data[batch_data>0], "\nbatch_calib:",batch_calib)
        
