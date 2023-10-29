from PIL import Image
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from SAM import create_mask

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

if __name__ == "__main__":
    img = cv2.imread('chicken_wound.jpg')
    cropped_img = img[1200:2200, 1700:2300]
    cv2.imwrite('cropped_wound.jpg', cropped_img)
    #img = cv2.imread('cropped_wound.jpg')
    mask, img = create_mask('cropped_wound.jpg', np.array([[349, 600]]), 'base')
    cv2.imwrite('sam_mask.jpg', mask)
    cv2.imwrite('sam_img.jpg', img)
    new_edge_detector = EdgeDetector()
    # edges = new_edge_detector.find_edges(mask)
    # cv2.imwrite('edges_sam.jpg', edges)
    mask = cv2.imread('sam_mask.jpg')
    img_dilated = new_edge_detector.dilate_to_line(mask, 20)
    #plt.subplot(121)
    #np.array([[349, 600]])
    #plt.imshow(img_dilated)
    #plt.show()
    cv2.imwrite("dilated_sam.jpg", img_dilated)

    skeleton = skeletonize(img_dilated)

    plt.imsave('skeleton_sam.png', skeleton)