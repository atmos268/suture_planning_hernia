#!/usr/bin/env python3
import cv2
import viser
import torch
import numpy as np
from scipy.spatial import cKDTree

DEVICE = "cuda"


class VisualizePointcloudNode:

    def __init__(self):
        self.chicken_number = 3
        self.server_ = viser.ViserServer()

        self.image_callback()

    def load_image(self, imfile):
        img = torch.from_numpy(imfile).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def image_callback(self):

        point_cloud_data = np.load(f"hernia_results/point_cloud{self.chicken_number}.npy")
        rgb_cloud_data = np.load(f"hernia_results/rgb_cloud{self.chicken_number}.npy")

        print("point_cloud_data shape is: ", np.shape(point_cloud_data))
        print("rgb_cloud_data shape is: ", np.shape(rgb_cloud_data))

        kdtree = cKDTree(point_cloud_data)
        self.insertion = np.load(f"hernia_results/insertion_pts{self.chicken_number}.npy")
        self.extraction = np.load(f"hernia_results/extraction_pts{self.chicken_number}.npy")
        self.centerline = np.load(f"hernia_results/line_pts_3d_{self.chicken_number}.npy")

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

        centerline_color = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (self.centerline.shape[0], 1))  # Yellow

        self.server_.add_point_cloud(
            "mesh", points=point_cloud_data, colors=rgb_cloud_data, point_size=0.005, point_shape="sparkle"
        )
        # self.server_.add_point_cloud(
        #     "insertion_points",
        #     points=insertion_pts_point_cloud,
        #     colors=insertion_points_color,
        #     point_size=0.01,
        #     point_shape="sparkle",
        # )
        # self.server_.add_point_cloud(
        #     "extraction_points",
        #     points=extraction_pts_point_cloud,
        #     colors=extraction_points_color,
        #     point_size=0.01,
        #     point_shape="sparkle",
        # )
        self.server_.add_point_cloud(
            "centerline_points",
            points=self.centerline,
            colors=centerline_color,
            point_size=0.005,
            point_shape="circle"
        )

        import pdb

        pdb.set_trace()
        exit()


if __name__ == "__main__":
    # Initialize as ROS node
    visualize_pointcloud_node = VisualizePointcloudNode()
