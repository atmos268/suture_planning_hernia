from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from point_ordering import get_pt_ordering
# import plantcv
from SAM import create_mask
from segment_anything import SamPredictor
from largestCC import keep_largest_connected_component
from fillHoles import fillHoles
'''This class will process an image, and produce a spline of where the wound is based on the image'''
class EdgeDetector:
    
    def find_edges(self, img):
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("grayscale_image.jpg", grayscale_image)
        # grayscale_image = np.clip(grayscale_image, 0, 170)
        cv2.imwrite("grayscale_image_clip.jpg", grayscale_image)
        # blur = cv2.bilateralFilter(grayscale_image, 5, 100, 150)
        cv2.imwrite("blur_clip.jpg", grayscale_image)
        return cv2.Canny(grayscale_image, 100, 600)

    def dilate_to_line(self, edge_mask, kernel_dim):
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
        return cv2.dilate(edge_mask, kernel, iterations=1) 
 
    def generate_spline(self, pixels):
        pass

# def keep_largest_connected_component(imgPath):
#     # Ensure the input image is in binary format (0 and 255).
#     input_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

#     print(len(np.unique(input_image)))
#     # Ensure the image is binary (threshold if needed).
#     _, binary_image = cv2.threshold(input_image, 128, 255, cv2.THRESH_BINARY)

#     print(len(np.unique(binary_image)))
#     # if len(np.unique(binary_image)) != 2:
#     #     raise ValueError("Input image must be binary (0 and 255).")

#     # Find connected components in the binary image.
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

#     # Find the index of the largest connected component (excluding the background).
#     largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

#     # Create a binary mask for the largest component.
#     largest_component_mask = (labels == largest_component_index).astype(np.uint8) * 255

#     return largest_component_mask


# mallika's code for point clicking

def click_points_simple(img):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img)
    left_coords,right_coords = [], []
    def onclick(event):
        xind,yind = int(event.xdata),int(event.ydata)
        coords=(xind,yind)
        nonlocal left_coords,right_coords
        if(event.button==1):
            left_coords.append(coords)
        elif(event.button==3):
            right_coords.append(coords)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return left_coords,right_coords

if __name__ == "__main__":

    # for testing puposes
    get_pt_ordering('binary_skeleton.npy')
    print("done")
    box_method = True

    img_path = 'image_left_001.png'

    # load the image and convert into
    # numpy array
    img = Image.open(img_path)
    
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = np.asarray(img)

    if box_method:
        left_coords, right_coords = click_points_simple(numpydata)
        if len(left_coords) != 2:
            raise ValueError("Please select 2 points (top left, bottom right)")
        
        # display box and image
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(numpydata)

        # Create a Rectangle patch
        w = left_coords[1][0] - left_coords[0][0]
        h = left_coords[1][1] - left_coords[0][1]
        rect = patches.Rectangle(tuple(left_coords[0]), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()

        box = left_coords[0] + left_coords[1]
        mask = create_mask(img_path, box, [], 'base', box_method=True)

        # make the mask into an image and save
        im = Image.fromarray(mask)
        im.save('sam_mask.jpg')

    else:

        left_coords, right_coords = click_points_simple(numpydata)
        print(left_coords, right_coords)

        cv2_img = cv2.imread(img_path)

        num_left = len(left_coords)
        num_right = len(right_coords)

        fore_back = [1 for _ in range(num_left)] + [0 for _ in range(num_right)]

        mask, img = create_mask(img_path, np.array(left_coords + right_coords), np.array(fore_back), 'base')
    
        cv2.imwrite('sam_mask.jpg', mask)
    mask = keep_largest_connected_component('sam_mask.jpg')
    cv2.imwrite('sam_mask.jpg', mask)
    # cv2.imwrite('sam_img.jpg', img)
    
    new_edge_detector = EdgeDetector()
    mask = cv2.imread('sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 5)
    cv2.imwrite("dilated_sam.jpg", img_dilated)
    img_dilated = fillHoles('dilated_sam.jpg')
    cv2.imwrite("filledHoles.jpg", img_dilated)

    # check for binary?
    
    # threshold
    binary_image = np.where(img_dilated > 0, 1, 0)
    skeleton = skeletonize(binary_image)

    np.save('binary_skeleton.npy', skeleton)

    plt.imshow(skeleton)
    plt.show()


    plt.imsave('skeleton_sam.jpg', skeleton)

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(numpydata)
    ax1.title.set_text("Original")

    img = Image.open("filledHoles.jpg")
    numpydata = np.asarray(img)
    ax2.imshow(numpydata)
    ax2.title.set_text("New Dilation (filled holes)")

    img = Image.open('skeleton_sam.jpg')
    numpydata = np.asarray(img)
    ax3.imshow(numpydata)
    ax3.title.set_text("Skeletonization")

    fig.tight_layout()

    # plot the image, dilation, skeleton
    plt.savefig('temp.jpg', dpi=1200)

    # now, try to order the points
    ordered_pts = get_pt_ordering('skeleton_sam.jpg')

