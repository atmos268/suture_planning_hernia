#!/usr/bin/env python3
import cv2
import viser
import torch
import numpy as np
from scipy.spatial import cKDTree

DEVICE = 'cuda'
class VisualizePointcloudNode:

    def __init__(self):
        self.chicken_number = 3
        self.server_ = viser.ViserServer()

        self.image_callback()
        

    def load_image(self,imfile):
        img = torch.from_numpy(imfile).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
    
    def image_callback(self):
        self.left_img = cv2.imread(f'dan_chicken/left_exp_00{self.chicken_number}.png')
        self.right_img = cv2.imread(f'dan_chicken/right_exp_00{self.chicken_number}.png')
        point_cloud_data = np.load(f'dan_point_cloud_data/point_cloud{self.chicken_number}.npy')
        rgb_cloud_data = np.load(f'dan_point_cloud_data/rgb_cloud{self.chicken_number}.npy')
        
        kdtree = cKDTree(point_cloud_data)
        self.insertion = np.load(f'dan_insertion_extraction_pts/insertion_pts{self.chicken_number}.npy')
        self.extraction = np.load(f'dan_insertion_extraction_pts/extraction_pts{self.chicken_number}.npy')
        
        insertion_pts_point_cloud = []
        for insertion_pt in self.insertion:
            _, idxs = kdtree.query(insertion_pt, 100)
            insertion_pts_point_cloud.extend(point_cloud_data[idxs].tolist())
        insertion_pts_point_cloud = np.array(insertion_pts_point_cloud)
        insertion_points_color = np.zeros_like(insertion_pts_point_cloud, dtype=np.uint8)
        insertion_points_color[:, 1] = 255

        extraction_pts_point_cloud = []
        for extraction_pt in self.extraction:
            _, idxs = kdtree.query(extraction_pt, 100)
            extraction_pts_point_cloud.extend(point_cloud_data[idxs].tolist())
        extraction_pts_point_cloud = np.array(extraction_pts_point_cloud)
        extraction_points_color = np.zeros_like(extraction_pts_point_cloud, dtype=np.uint8)
        extraction_points_color[:, 0] = 255

        self.server_.add_point_cloud('mesh',points=point_cloud_data,colors=rgb_cloud_data,point_size=0.001,point_shape='sparkle')
        self.server_.add_point_cloud('insertion_points',points=insertion_pts_point_cloud,colors=insertion_points_color,point_size=0.001,point_shape='sparkle')
        self.server_.add_point_cloud('extraction_points',points=extraction_pts_point_cloud,colors=extraction_points_color,point_size=0.001,point_shape='sparkle')
        import pdb
        pdb.set_trace()
        exit()

if __name__ == '__main__':
    # Initialize as ROS node
    visualize_pointcloud_node = VisualizePointcloudNode()