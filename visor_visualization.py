#!/usr/bin/env python3
import sys
# sys.path.append('/home/kushtimusprime/RAFT-Stereo/core')
# from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import os
import cv2
import time
from matplotlib import cm
import pyautogui
import viser
import numpy as np
import torch
from cv_bridge import CvBridge
import scipy.interpolate as inter
from matplotlib import pyplot as plt
from SuturePlacer import SuturePlacer
from sensor_msgs.msg import PointCloud,Image
from autolab_core import RigidTransform, ColorImage, DepthImage,CameraIntrinsics
from skimage.morphology import skeletonize
from geometry_msgs.msg import PointStamped, Point32
# from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import argparse

DEVICE = 'cuda'
class VisualizePointcloudNode:

    def __init__(self):
        self.ready = False
        # self.filepath_ = os.path.dirname(os.path.abspath(__file__))
        # sam_checkpoint = os.path.join(self.filepath_,"segment_anything/sam_vit_h_4b8939.pth")
        sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        device = "cpu"

        sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
        sam.to(device=device)

        self.predictor = SAM2ImagePredictor(sam)
        print("SAM Initialized")
        self.filepath_ = os.path.dirname(os.path.abspath(__file__))
        self.q_stereo_matrix_ = np.load(os.path.join(self.filepath_,'../../calibration_config/allied_vision_matrices/Q_STEREO_mat.npy'))
        self.f_ = self.q_stereo_matrix_[2][3]
        self.cx_ = -self.q_stereo_matrix_[0][3]
        self.cy_ = -self.q_stereo_matrix_[1][3]
        self.Tx_ = -1/self.q_stereo_matrix_[3][2]
        self.bridge_ = CvBridge()
        
        self.padder_ = InputPadder(torch.empty(1,3,960,1280).shape, divis_by=32)
        self.parser_ = argparse.ArgumentParser()
        self.debug_ = False
        ################################################### fastraft ###################################################
        self.parser_.add_argument(
            "--restore_ckpt",
            default="/home/kushtimusprime/RAFT-Stereo/models/raftstereo-middlebury.pth",
            help="restore checkpoint",
        )
        self.parser_.add_argument(
            "--save_numpy",
            action="store_true",
            help="save output as numpy arrays"
        )
        self.parser_.add_argument(
            "-l",
            "--left_imgs",
            help="path to all first (left) frames",
            default='/home/kushtimusprime/RAFT-Stereo/2024_03_26_data_raft/*/im0.png',
        )
        self.parser_.add_argument(
            "-r",
            "--right_imgs",
            help="path to all second (right) frames",
            default='/home/kushtimusprime/RAFT-Stereo/2024_03_26_data_raft/*/im1.png',
        )
        self.parser_.add_argument(
            "--mask_imgs",
            help="path to all mask frames",
            default='/home/kushtimusprime/RAFT-Stereo/2024_03_26_data_raft/*/mask.png',
        )
        self.parser_.add_argument(
            "--output_directory", help="directory to save output", default="demo_output"
        )
        self.parser_.add_argument(
            "--shared_backbone",
            action="store_true",
            default=False,
            help="use a single backbone for the context and feature encoders",
        )
        self.parser_.add_argument(
            "--mixed_precision",
            action="store_true",
            default=False,
            help="use mixed precision",
        )
        self.parser_.add_argument(
            "--slow_fast_gru",
            action="store_true",
            default=False,
            help="iterate the low-res GRUs more frequently",
        )
        self.parser_.add_argument(
            "--corr_implementation",
            default="reg",
            choices=["reg", "alt", "reg_cuda", "alt_cuda"],
            help="correlation volume implementation",
        )
        self.parser_.add_argument(
            "--context_norm",
            default="batch",
            choices=["group", "batch", "instance", "none"],
            help="normalization of context encoder",
        )
        # Architecture choices
        self.parser_.add_argument(
            "--valid_iters",
            type=int,
            default=64,
            help="number of flow-field updates during forward pass",
        )
        self.parser_.add_argument(
            "--hidden_dims",
            nargs="+",
            type=int,
            default=[128] * 3,
            help="hidden state and context dimensions",
        )
        self.parser_.add_argument(
            "--corr_levels",
            type=int,
            default=4,
            help="number of levels in the correlation pyramid",
        )
        self.parser_.add_argument(
            "--corr_radius",
            type=int,
            default=4,
            help="width of the correlation pyramid",
        )
        self.parser_.add_argument(
            "--n_downsample",
            type=int,
            default=2,
            help="resolution of the disparity field (1/2^K)",
        )
        self.parser_.add_argument(
            "--n_gru_layers",
            type=int,
            default=3,
            help="number of hidden GRU levels"
        )
        ################################################## fastraft ###################################################

        self.args_ = self.parser_.parse_args()
        self.model_ = torch.nn.DataParallel(RAFTStereo(self.args_), device_ids=[0])
        self.model_.load_state_dict(torch.load(self.args_.restore_ckpt))
        self.image_data_ = None
        self.model_ = self.model_.module
        self.model_.to(DEVICE)
        self.model_.eval()
        self.camera_intrinsics_ = CameraIntrinsics(
            frame="av_left_frame", fx=self.f_, fy=self.f_, cx=self.cx_, cy=self.cy_
        )
        self.server_ = viser.ViserServer()
        gui_reset_up = self.server_.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = viser.transforms.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

        self.wound_slit_area = None
        self.raised_wound = None
        self.full_phantom = None
        self.ready = True
        self.debug = False
        left_img = cv2.imread('/home/kushtimusprime/stitch_ros_ws/src/ros_raft/sample_data_3/left_img.png')
        right_img = cv2.imread('/home/kushtimusprime/stitch_ros_ws/src/ros_raft/sample_data_3/right_img.png')
        import pdb
        pdb.set_trace()
        self.image_callback(left_img,right_img)
        
        
    def get_segmentation_masks(self,image):
        suture_plan_output_folder = os.path.join(self.filepath_,'../suture_plan_output')
        if not os.path.exists(suture_plan_output_folder):
            os.makedirs(suture_plan_output_folder)
        wound_slit_area_filepath = os.path.join(suture_plan_output_folder,'wound_slit_area.png')
        if os.path.isfile(wound_slit_area_filepath):
            self.wound_slit_area = cv2.imread(wound_slit_area_filepath,0)
        else:
            while self.wound_slit_area is None:
                self.wound_slit_area = self.resegment_with_sam('wound_slit_area',image)
            cv2.imwrite(wound_slit_area_filepath,255 *self.wound_slit_area)
        raised_wound_filepath = os.path.join(suture_plan_output_folder,'raised_wound.png')
        if os.path.isfile(raised_wound_filepath):
            self.raised_wound = cv2.imread(raised_wound_filepath,0)
        else:
            while self.raised_wound is None:
                self.raised_wound = self.resegment_with_sam('raised_wound',image)
            cv2.imwrite(raised_wound_filepath,255 * self.raised_wound)
        full_phantom_filepath = os.path.join(suture_plan_output_folder,'full_phantom.png')
        if os.path.isfile(full_phantom_filepath):
            self.full_phantom = cv2.imread(full_phantom_filepath,0)
        else:
            while self.full_phantom is None:
                self.full_phantom = self.resegment_with_sam('full_phantom',image)
            cv2.imwrite(full_phantom_filepath,255 * self.full_phantom) 
        
    
    def resize_to_screen(self,image, screen_width, screen_height):
        h, w = image.shape[:2]  # Original image height and width
        aspect_ratio = w / h

        # If the image width exceeds the screen width, resize based on width
        if w > screen_width:
            w = screen_width
            h = int(w / aspect_ratio)
        
        # If the image height exceeds the screen height after width adjustment, resize based on height
        if h > screen_height:
            h = screen_height
            w = int(h * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (w, h))

        # Calculate scaling factors based on original dimensions
        scaling_factor_x = w / image.shape[1]  # Scale factor for width
        scaling_factor_y = h / image.shape[0]  # Scale factor for height

        return resized_image, scaling_factor_x, scaling_factor_y

    def resegment_with_sam(self,segment_area,image):
        self.predictor.set_image(image)
        clicked_points = []

        # Get the screen resolution
        screen_width, screen_height = pyautogui.size()
        screen_width -= 100
        screen_height -= 200

        # Resize the image to fit the screen and calculate scaling factors
        resized_image, scaling_factor_x, scaling_factor_y = self.resize_to_screen(image, screen_width, screen_height)
        # Mouse callback function to capture clicks
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Append clicked point to the list
                clicked_points.append((int(x / scaling_factor_x), int(y / scaling_factor_y)))  # Scale to original image
                print(f"Clicked point: ({x}, {y}) -> Original point: ({int(x / scaling_factor_x)}, {int(y / scaling_factor_y)})")
                
                # Optionally draw a circle on the image where clicked
                cv2.circle(resized_image, (x, y), 1, (0, 255, 0), -1)  # Draw a green circle
                cv2.imshow(segment_area, resized_image)

        # Create a named window and set the mouse callback
        cv2.namedWindow(segment_area)
        cv2.setMouseCallback(segment_area, click_event)

        # Display the resized masked image
        while True:
            cv2.imshow(segment_area, resized_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Press 'c' to clear clicked points
                clicked_points.clear()  # Clear the list of clicked points
                resized_image = cv2.resize(image, (resized_image.shape[1], resized_image.shape[0]))  # Reset the displayed image
                cv2.imshow(segment_area, resized_image)
                print("Clicked points cleared.")
            elif key == 13:  # Press 'Return' to exit and print points
                break
        # closing all open windows
        cv2.destroyWindow(segment_area)

        # Print the list of clicked points
        print("List of clicked points (original image coordinates):", clicked_points)
        clicked_points_np = np.array(clicked_points)
        input_label = np.array(clicked_points_np.shape[0] * [1])
        masks, scores, logits = self.predictor.predict(point_coords=clicked_points_np,
        point_labels=input_label,multimask_output=True)
        row1 = cv2.hconcat([image, cv2.cvtColor(255 * masks[0,:,:].astype(np.uint8),cv2.COLOR_GRAY2BGR)])
        # Concatenate the last two images horizontally
        row2 = cv2.hconcat([cv2.cvtColor(255 * masks[1,:,:].astype(np.uint8),cv2.COLOR_GRAY2BGR), cv2.cvtColor(255 * masks[2,:,:].astype(np.uint8),cv2.COLOR_GRAY2BGR)])

        # Now concatenate the two rows vertically to form a 2x2 grid
        grid_image = cv2.vconcat([row1, row2])

        resized_grid_image, scaling_factor_x, scaling_factor_y = self.resize_to_screen(grid_image, screen_width, screen_height)
        cv2.imshow('0 is TR, 1 is BL, 2 is BR', resized_grid_image)
        final_mask = None  # Variable to store the selected mask

        # Wait for user keypress
        while True:
            key = cv2.waitKey(0)  # Wait indefinitely for a keypress

            # Check which key was pressed
            if key == ord('0'):  # If '0' is pressed
                final_mask = masks[0]
                print("Mask 0 selected.")
                break
            elif key == ord('1'):  # If '1' is pressed
                final_mask = masks[1]
                print("Mask 1 selected.")
                break
            elif key == ord('2'):  # If '2' is pressed
                final_mask = masks[2]
                print("Mask 2 selected.")
                break
        cv2.destroyWindow('0 is TR, 1 is BL, 2 is BR')
        final_mask = 255 * final_mask.astype(np.uint8)
        blue_mask = np.zeros_like(image)
        blue_mask[:, :, 0] = final_mask  # Set the blue channel to the mask values
        # Overlay the blue mask on the original image
        overlayed_image = cv2.addWeighted(image, 1.0, blue_mask, 0.5, 0)

        # Resize the combined image to fit the screen
        resized_overlayed_image,_,_ = self.resize_to_screen(overlayed_image, screen_width, screen_height)

        cv2.imshow('Right arrow if good', resized_overlayed_image)
        # Wait for a keypress
        key = cv2.waitKey(0)  # Wait indefinitely for a keypress

        # Check for the right arrow key (key code for right arrow is 83)
        if key == 83:  # Right arrow key pressed
            print("Right arrow pressed. Returning final_mask.")
            cv2.destroyWindow('Right arrow if good')
            result = 255 * final_mask.astype(np.uint8)  # Return final_mask
        else:
            print("Other key pressed. Returning None.")
            cv2.destroyWindow('Right arrow if good')
            result = None  # Return None
        return result

    def load_image(self,imfile):
        img = torch.from_numpy(imfile).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
    
    def getDepthImage(self,left_img,right_img):
        
        # Follows RAFT Stereo demo code for obtaining disparity image
        image1 = self.load_image(left_img)
        image2 = self.load_image(right_img)
        image1, image2 = self.padder_.pad(image1, image2)
        _, flow_up = self.model_(image1, image2, iters=self.args_.valid_iters, test_mode=True)
        flow_up = self.padder_.unpad(flow_up).squeeze()

        flow_up_np = -flow_up.detach().cpu().numpy().squeeze()
        depth_image = (self.f_ * self.Tx_) / abs(flow_up_np)
        return depth_image
    
    def rgbd_image_to_rgb_pointcloud(self, color_image, depth_image, camera_intrinsics,mask=None):
        if mask is None:
            color_image_masked = color_image.data
            depth_image_masked = depth_image.data
        else:
            color_image_masked = cv2.bitwise_and(color_image.data,color_image.data,mask=mask)
            depth_image_masked = cv2.bitwise_and(depth_image.data,depth_image.data,mask=mask)
        rgb_cloud_data = color_image_masked.reshape(-1, 3)
        point_cloud_data = camera_intrinsics.deproject(DepthImage(data=depth_image_masked,frame='av_left_frame')).data.T
        non_zero_indices = np.all(point_cloud_data != [0, 0, 0], axis=-1)
        rgb_cloud_data = rgb_cloud_data[non_zero_indices]
        point_cloud_data = point_cloud_data[non_zero_indices]
        return rgb_cloud_data, point_cloud_data
    
    def fit_plane(self, points):
        ## plot point cloud
        ## average and center points
        P = points
        P_mean = np.mean(P, axis=0)
        ransac_iters = 1000
        largest_inlier_count, largest_inlier_set = 0, np.array([])
        inlier_epsilon = 0.0005
        for _ in range(ransac_iters):
            # We sample 100 points each time to fit the plane
            #TODO: Experiment with seeing if we can drop this number
            # Should make our algo more efficient, but makes us more sensitive to noise

            sampled_pts = P[np.random.choice(len(P), min(len(P),100)), :]
            sampled_pts = np.vstack((sampled_pts, np.array(P_mean)))

            ## solve for Ax + By + Cz = D
            ## -(A/C)x + -(B/C)y + (D/C) = z
            M = np.hstack(
                (sampled_pts[:, :2], np.ones((len(sampled_pts), 1)))
            )  # matrix of [xs, ys, 1]
            v = sampled_pts[:, 2]  # vector of [zs]
            soln = np.linalg.lstsq(M, v, rcond=None)[0]  # [-A, -B, D]
            A, B, C, D = -soln[0], -soln[1], 1, soln[2]
            distances = A * P[:, 0] + B * P[:, 1] + C * P[:, 2] - D
            distances = np.divide(distances, np.sqrt(A**2 + B**2 + C**2))
            inliers = np.where(np.abs(distances) < inlier_epsilon)
            num_inliers = len(inliers[0])
            if num_inliers > largest_inlier_count:
                largest_inlier_count = num_inliers
                largest_inlier_set = P[inliers]
        
        # Return plane with the most inliers along with corresponding inlier points
        inlier_points = largest_inlier_set
        M = np.hstack((inlier_points[:, :2], np.ones((len(inlier_points), 1))))
        v = inlier_points[:, 2]
        soln = np.linalg.lstsq(M, v, rcond=None)[0]
        A, B, C, D = -soln[0], -soln[1], 1, soln[2]
        return np.array([A, B, C, D]), inlier_points
    
    def skeletonize_detection(self, needle_detection_mask):
        self.detection_proba_threshold = 0.5
        needle_detection_mask_binary = np.where(
            needle_detection_mask > self.detection_proba_threshold, 1.0, 0.0
        )
        needle_detection_mask_binary_skeleton = skeletonize(
            needle_detection_mask_binary
        )
        needle_detection_mask_skeleton = np.where(
            needle_detection_mask_binary_skeleton, needle_detection_mask, 0.0
        )
        return needle_detection_mask_skeleton
    
    def project_points_onto_plane(self, points, plane):
        # Has visual diagram of why this works: https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d

        A, B, C, D = plane
        normal_vec = np.array([A, B, C])
        P_mean = np.mean(points, axis=0)
        # Calculate the z-coordinate of the mean point when projected onto the plane
        P_mean_z = -A * (P_mean[0]) + -B * (P_mean[1]) + D

        # Define a point Q on the plane using the x and y coordinates of the mean point and the z-coordinate calculated above
        Q = np.array([P_mean[0], P_mean[1], P_mean_z])

        # Compute the vector from each point in the input cloud to the point Q on the plane
        PQ = points - Q

        # Calculate the dot product of each vector PQ with the normal vector of the plane
        dot_products = np.sum(PQ * normal_vec, axis=1)

        # Project each point onto the plane using the dot products and the normal vector
        projected_pts = (
            points
            - (dot_products / np.linalg.norm(normal_vec) ** 2)[:, np.newaxis]
            * normal_vec
        )
        return projected_pts
    def image_callback(self,left_image,right_image):
        if self.ready:
            self.left_img = left_image #self.bridge_.imgmsg_to_cv2(left_image_msg,desired_encoding='bgr8')
            self.right_img = right_image #self.bridge_.imgmsg_to_cv2(right_image_msg,desired_encoding='bgr8')
            self.get_segmentation_masks(self.left_img)
            full_rgb_cloud,full_point_cloud = self.make_full_pointcloud(self.left_img,self.right_img,'full_pointcloud')
            wound_mask_skeletonized = self.skeletonize_detection(self.wound_slit_area)
            wound_mask_skeletonized_rgb_cloud,wound_mask_skeletonized_point_cloud = self.make_segmented_pointclouds(self.left_img,self.right_img,wound_mask_skeletonized.astype(np.uint8),'wound_center_skeletonized')
            wound_center_rgb_cloud,wound_center_point_cloud = self.make_segmented_pointclouds(self.left_img,self.right_img,self.wound_slit_area,'wound_center')
            wound_top = cv2.dilate(self.wound_slit_area,np.ones((5, 5), np.uint8),iterations=6)
            wound_top = wound_top - self.wound_slit_area
            wound_top_rgb_cloud,wound_top_point_cloud =self.make_segmented_pointclouds(self.left_img,self.right_img,wound_top,'wound_top')
            phantom_not_wound = self.full_phantom - self.raised_wound
            phantom_not_wound_rgb_cloud,phantom_not_wound_point_cloud =self.make_segmented_pointclouds(self.left_img,self.right_img,phantom_not_wound,'phantom_not_wound')
            wound_rgb_cloud,wound_point_cloud =self.make_segmented_pointclouds(self.left_img,self.right_img,self.raised_wound,'wound')
            phantom_rgb_cloud,phantom_point_cloud =self.make_segmented_pointclouds(self.left_img,self.right_img,self.full_phantom,'phantom')
            wound_top_plane, wound_top_inlier_points = self.fit_plane(wound_top_point_cloud)
            A_wound_top, B_wound_top, C_wound_top, D_wound_top = wound_top_plane
            if self.debug:
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(
                    wound_top_point_cloud[:, 0],
                    wound_top_point_cloud[:, 1],
                    wound_top_point_cloud[:, 2],
                    c="k",
                    alpha=0.2,
                )
                xs, ys, zs = wound_top_inlier_points[:, 0], wound_top_inlier_points[:, 1], wound_top_inlier_points[:, 2]
                xx, yy = np.meshgrid(
                    np.linspace(np.min(xs), np.max(xs), 10),
                    np.linspace(np.min(ys), np.max(ys), 10),
                )
                zz = -A_wound_top * (xx) + -B_wound_top * (yy) + D_wound_top
                ax.plot_surface(xx, yy, zz, cmap=cm.Greens, alpha=0.5)
            phantom_not_wound_plane, phantom_not_wound_inlier_points = self.fit_plane(phantom_not_wound_point_cloud)
            A_phantom_not_wound, B_phantom_not_wound, C_phantom_not_wound, D_phantom_not_wound = phantom_not_wound_plane
            if self.debug:
                ax = plt.figure().add_subplot(projection="3d")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.scatter(
                    phantom_not_wound_point_cloud[:, 0],
                    phantom_not_wound_point_cloud[:, 1],
                    phantom_not_wound_point_cloud[:, 2],
                    c="k",
                    alpha=0.2,
                )
                xs, ys, zs = phantom_not_wound_inlier_points[:, 0], phantom_not_wound_inlier_points[:, 1], phantom_not_wound_inlier_points[:, 2]
                xx, yy = np.meshgrid(
                    np.linspace(np.min(xs), np.max(xs), 10),
                    np.linspace(np.min(ys), np.max(ys), 10),
                )
                zz = -A_phantom_not_wound * (xx) + -B_phantom_not_wound * (yy) + D_phantom_not_wound
                ax.plot_surface(xx, yy, zz, cmap=cm.Greens, alpha=0.5)
                plt.show()
            wound_mask_skeletonized_plane_rgb_cloud = wound_mask_skeletonized_rgb_cloud
            wound_mask_skeletonized_plane_point_cloud = self.project_points_onto_plane(wound_mask_skeletonized_point_cloud,wound_top_plane)
            self.server_.add_point_cloud('wound_mask_skeletonized_plane', points=wound_mask_skeletonized_plane_point_cloud, colors=wound_mask_skeletonized_plane_rgb_cloud, point_size=0.0001)
            wound_line,wound_lier_inlier_mask = self.ransac_line_fitting(wound_mask_skeletonized_plane_point_cloud)
            wound_line_inliers = wound_mask_skeletonized_plane_point_cloud[wound_lier_inlier_mask]
            wound_line_projected_points = self.project_points_onto_line(wound_line_inliers,*wound_line)
            sorted_wound_line_projected_points = wound_line_projected_points[wound_line_projected_points[:,2].argsort()]
            wound_center_points = np.linspace(sorted_wound_line_projected_points[0],sorted_wound_line_projected_points[-1],6)
            self.server_.add_point_cloud('wound_center_points',points=wound_center_points,colors=np.ones_like(wound_center_points),point_size=0.001,point_shape='sparkle')
            wound_top_normal = np.array([A_wound_top, B_wound_top, C_wound_top])
            wound_top_normal = wound_top_normal / np.linalg.norm(wound_top_normal)
            if(wound_top_normal[2] < 0):
                wound_top_normal = -wound_top_normal
            wound_line_vector = wound_line[1]
            wound_line_vector = wound_line_vector / np.linalg.norm(wound_line_vector)
            if(wound_line_vector[2] < 0):
                wound_line_vector = -wound_line_vector
            wound_width_vector = np.cross(wound_top_normal,wound_line_vector)
            wound_width_vector = wound_width_vector / np.linalg.norm(wound_width_vector)
            wound_width_projections = np.dot(wound_point_cloud,wound_width_vector)
            wound_width = np.max(wound_width_projections) - np.min(wound_width_projections)
            insertion_points_x = wound_center_points + (wound_width_vector * (wound_width / 2))
            insertion_points_color = np.zeros_like(insertion_points_x, dtype=np.uint8)
            insertion_points_color[:, 1] = 255
            extraction_points_x = wound_center_points - (wound_width_vector * (wound_width / 2))
            extraction_points_color = np.zeros_like(extraction_points_x, dtype=np.uint8)
            extraction_points_color[:, 0] = 255
            wound_height = D_phantom_not_wound - D_wound_top
            insertion_points_final = insertion_points_x + (wound_top_normal * (wound_height / 4))
            insertion_points_color = np.zeros_like(insertion_points_final, dtype=np.uint8)
            insertion_points_color[:, 1] = 255
            self.server_.add_point_cloud('insertion_points_final',points=insertion_points_final,colors=insertion_points_color,point_size=0.001,point_shape='sparkle')
            extraction_points_final = extraction_points_x + (wound_top_normal * (wound_height / 4))
            extraction_points_color = np.zeros_like(extraction_points_final, dtype=np.uint8)
            extraction_points_color[:, 0] = 255
            self.server_.add_point_cloud('extraction_points_final',points=extraction_points_final,colors=extraction_points_color,point_size=0.001,point_shape='sparkle')
            pre_insertion_points_final = insertion_points_final + (wound_width_vector * 0.002)
            pre_insertion_points_color = np.zeros_like(pre_insertion_points_final, dtype=np.uint8)
            pre_insertion_points_color[:, 2] = 255
            self.server_.add_point_cloud('pre_insertion_points_final',points=pre_insertion_points_final,colors=pre_insertion_points_color,point_size=0.001,point_shape='sparkle')
            import pdb
            pdb.set_trace()
            suture_plan_output_folder = os.path.join(self.filepath_,'../suture_plan_output')
            np.save(os.path.join(suture_plan_output_folder,'center_points_cam_frame.npy'),wound_center_points)
            np.save(os.path.join(suture_plan_output_folder,'insertion_points_cam_frame.npy'),insertion_points_final)
            np.save(os.path.join(suture_plan_output_folder,'extraction_points_cam_frame.npy'),extraction_points_final)
            np.save(os.path.join(suture_plan_output_folder,'pre_insertion_points_cam_frame.npy'),pre_insertion_points_final)
            exit()


    def project_points_onto_line(self,points, point_on_line, direction):
        """
        Projects points onto a line defined by `point_on_line` and `direction`.
        
        Args:
            points (numpy.ndarray): Nx3 array of points to project.
            point_on_line (numpy.ndarray): A point on the line (1x3).
            direction (numpy.ndarray): The direction vector of the line (1x3).
            
        Returns:
            projected_points (numpy.ndarray): Nx3 array of points projected onto the line.
        """
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Compute projections
        projected_points = point_on_line + np.dot(points - point_on_line, direction)[:, np.newaxis] * direction
        
        return projected_points
    def ransac_line_fitting(self,points, threshold=0.0001, max_iterations=1000):
        """
        Fits a line to a 3D point cloud using RANSAC.
        
        Args:
            points (numpy.ndarray): Nx3 array of 3D points.
            threshold (float): Distance threshold to count as an inlier.
            max_iterations (int): Number of iterations to run RANSAC.

        Returns:
            best_line (tuple): (point_on_line, direction_vector) of the best-fit line.
            inliers (numpy.ndarray): Boolean mask of inliers.
        """
        num_points = points.shape[0]
        best_inlier_count = 0
        best_line = None
        best_inliers = None

        for _ in range(max_iterations):
            # Randomly sample two points
            idx = np.random.choice(num_points, 2, replace=False)
            p1, p2 = points[idx]

            # Define the candidate line
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)  # Normalize
            point_on_line = p1

            # Compute distances of all points to the line
            distances = np.linalg.norm(
                np.cross(points - point_on_line, direction), axis=1
            ) / np.linalg.norm(direction)

            # Count inliers
            inliers = distances < threshold
            inlier_count = np.sum(inliers)

            # Update the best model if current one is better
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_line = (point_on_line, direction)
                best_inliers = inliers

        return best_line, best_inliers

    def make_segmented_pointclouds(self,left_img,right_img,mask,pointcloud_name):
        left_img = cv2.cvtColor(left_img,cv2.COLOR_RGB2BGR)
        right_img = cv2.cvtColor(right_img,cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            depth_image = self.getDepthImage(left_img,right_img)
            depth_image = -depth_image
            rgb_autolab_core = ColorImage(data=left_img, frame="av_left_frame")
            depth_autolab_core = DepthImage(data=depth_image, frame="av_left_frame")
            rgb_cloud_data, point_cloud_data = self.rgbd_image_to_rgb_pointcloud(
                rgb_autolab_core, depth_autolab_core, self.camera_intrinsics_,mask
            )
            self.server_.add_point_cloud(pointcloud_name, points=point_cloud_data, colors=rgb_cloud_data, point_size=0.0001)
            return rgb_cloud_data,point_cloud_data
    def make_full_pointcloud(self,left_img,right_img,pointcloud_name='full_pointcloud'):
        left_img = cv2.cvtColor(left_img,cv2.COLOR_RGB2BGR)
        right_img = cv2.cvtColor(right_img,cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            depth_image = self.getDepthImage(left_img,right_img)
            depth_image = -depth_image
            rgb_autolab_core = ColorImage(data=left_img, frame="av_left_frame")
            depth_autolab_core = DepthImage(data=depth_image, frame="av_left_frame")
            rgb_cloud_data, point_cloud_data = self.rgbd_image_to_rgb_pointcloud(
                rgb_autolab_core, depth_autolab_core, self.camera_intrinsics_,mask=None
            )
            self.server_.add_point_cloud(pointcloud_name, points=point_cloud_data, colors=rgb_cloud_data, point_size=0.0001)
            return rgb_cloud_data,point_cloud_data

if __name__ == '__main__':
    # Initialize as ROS node
    visualize_pointcloud_node = VisualizePointcloudNode()