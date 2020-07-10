import cv2
import os

images = []
images_names = []

set_names = ['Set1', 'Set2', 'Set3', 'TestSet1', 'TestSet2', 'TestSet3', 'TestSet4']
# set_nums = [3, 3, 8, 4, 9, 3, 5]
set_nums = [2, 2, 2, 2, 2, 2, 2]
suffix = 'rasnac'

for i, set_name in enumerate(set_names):
    images = []
    for j in range(int(set_nums[i])):
        images.append(cv2.imread(set_name + '_' + str(j + 1) + '_' + suffix + '.png'))
    cv2.imwrite('result_' + set_name + '_' + suffix + '.png', cv2.hconcat(images))



# for i, set_name in enumerate(set_names):
#     images = []
#     images.append(cv2.imread(set_name + '_1_' + set_name + '_2_' + suffix + '.png'))
#     images.append(cv2.imread(set_name + '_2_' + set_name + '_3_' + suffix + '.png'))
#
#     cv2.imwrite('result_' + set_name + '_' + suffix + '.png', cv2.hconcat(images))




