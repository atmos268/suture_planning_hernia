import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImagePointPicker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.points = []

    def click_event(self, event):
        x, y = event.xdata, event.ydata
        self.points.append((x, y))
        plt.plot(x, y, 'ro')
        plt.draw()

    def pick_points(self):
        img = mpimg.imread(self.image_path)
        plt.imshow(img)
        plt.title("Click two points on the image")
        plt.connect('button_press_event', self.click_event)
        plt.show()

        if len(self.points) == 2:
            return self.points
        else:
            print("You need to select two points on the image.")
            return None

# Example usage:
if __name__ == '__main__':
    img1_path = 'laceration-wound-caused-by-severe-260nw-624361613.jpeg'  # Replace with the path to your image
    picker = ImagePointPicker(img1_path)
    points = picker.pick_points()

    import numpy as np
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    from segment_anything import sam_model_registry, SamPredictor

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor1 = SamPredictor(sam)
    img1 = cv2.imread(img1_path)

    predictor1.set_image(img1)
    input_point = np.array([(85.63636363636363, 133.1363636363636), (135.6363636363636, 51.318181818181756)])
    input_label = np.array([0,1])
    masks_left, scores_left, logits_left = predictor1.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )


    masks_left = np.transpose(masks_left, (1, 2, 0))
    masked_left_img = img1*masks_left
    zero_one_mask = np.ones_like(masks_left, dtype=int) * masks_left * 256


    print(zero_one_mask)
    print(img1)
    def overlay_mask_on_image(original_image, mask, transparency=0.5):
        # Ensure that the original image and mask have the same dimensions
        if original_image.shape[:2] != mask.shape[:2]:
            raise ValueError("The original image and mask must have the same dimensions.")

        # Create a copy of the original image
        result = original_image.copy()

        # Convert the mask to 3 channels (BGR) and apply transparency
        mask_bgr = np.ones_like(original_image)  # Create a black mask with the same shape as the original image
        mask_bgr[:, :, 0] = 0  # Set blue channel to 0 (black)
        mask_bgr[:, :, 1] = 255  # Set green channel to 255 (yellow)
        mask_bgr[:, :, 2] = 0  # Set red channel to 0 (black)
        overlay = cv2.addWeighted(result, 1 - transparency, mask_bgr, transparency, 0)

        return overlay


    plt.subplot(122)
    #plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap='gray') # I would add interpolation='none'
    plt.imshow(masks_left) 
    plt.show()

