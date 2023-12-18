import subprocess
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np


left_image_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/left_image_1.png'
left_label_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/left_label_1.png'
right_image_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/right_image_1.png'
right_label_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/right_lable_1.png'

left_image = cv2.imread(left_image_path)
left_label = cv2.imread(left_label_path)
right_image = cv2.imread(right_image_path)
right_label = cv2.imread(right_label_path)

left_label = left_label*255
left_mask_image_psm1 = left_label[:,:,1]
left_mask_image_psm2 = left_label[:,:,2]
left_mask_image_needle = left_label[:,:,0]
left_label_gray = cv2.cvtColor(left_label, cv2.COLOR_BGR2GRAY)
left_masked = cv2.bitwise_and(left_image, left_image, mask=left_label_gray)
cv2.imwrite('left_label_gray.png', left_label_gray)
cv2.imwrite('left_masked.png', left_masked)
cv2.imwrite('left_original.png', left_image)

right_label = right_label*255
right_mask_image_psm1 = right_label[:,:,1]
right_mask_image_psm2 = right_label[:,:,2]
right_mask_image_needle = right_label[:,:,0]
right_label_gray = cv2.cvtColor(right_label, cv2.COLOR_BGR2GRAY)
right_masked = cv2.bitwise_and(right_image, right_image, mask=right_label_gray)
cv2.imwrite('right_label_gray.png', right_label_gray)
cv2.imwrite('right_masked.png', right_masked)
cv2.imwrite('right_original.png', right_image)

# plt.subplot(121)
# plt.title('right_label')
# plt.imshow(right_label)

# plt.subplot(122)
# plt.title('right_label_gray')
# plt.imshow(right_label_gray)

# plt.show()




# 시각화
# plt.subplot(231)
# plt.title('left_image')
# plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))

# plt.subplot(232)
# plt.title('left_label')
# plt.imshow(left_label)

# plt.subplot(233)
# plt.title('left_masked')
# plt.imshow(left_masked)

# plt.subplot(234)
# plt.title('right_image')
# plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))

# plt.subplot(235)
# plt.title('right_label')
# plt.imshow(right_label)

# plt.subplot(236)
# plt.title('right_masked')
# plt.imshow(right_masked)
# plt.show()


# 실행할 파이썬 스크립트 파일 경로
python_path = '/bin/python3'


# script_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/demo3.py'
# test = subprocess.run([
#     python_path, script_path,
#     "--restore_ckpt", "models/raftstereo-middlebury.pth",
#     "--corr_implementation", "alt",
#     "--mixed_precision",
#     "-l=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/data_new/left_001.png",
#     "-r=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/data_new/right_001.png"
# ])



script_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/demo.py'
test = subprocess.run([
    python_path, script_path,
    "--restore_ckpt", "models/raftstereo-middlebury.pth",
    "--corr_implementation", "alt",
    "--mixed_precision",
    "-l=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/left_image_1.png",
    "-r=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/right_image_1.png"
])

# script_path = '/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/demo20231029.py'
# test = subprocess.run([
#     python_path, script_path,
#     "--restore_ckpt", "models/raftstereo-middlebury.pth",
#     "--corr_implementation", "alt",
#     "--mixed_precision",
#     "-l=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/left_masked.png",
#     "-r=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/right_masked.png"
#     # "-l=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/left_11.png",
#     # "-r=/home/kimhs/Desktop/dVRK_Kimhs/4_RAFT/RAFT-Stereo-main/test_data/right_11.png"
# ])