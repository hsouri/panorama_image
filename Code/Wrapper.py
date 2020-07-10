#!/usr/bin/evn python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano

Author:

Hossein Souri (hsouri@umiacs.umd.edu)
PhD Student in Computer Vision and Machine Learning
University of Maryland, College Park
Note: Reference For blending function:
https://stackoverflow.com/
"""

# Code starts here:

import numpy as np
import cv2
import os
import copy
import sys
import matplotlib.pyplot as plt



def plot_corners(image, corner_point_vec, name, suffix=''):
	_image = copy.deepcopy(image)
	for point in corner_point_vec:
		x, y = point[0]
		cv2.circle(_image, (x, y), 4, (0, 0, 255), -1)
	cv2.imwrite(name + '_' + suffix + '.png', _image)

def corner_detection(images, images_names, plot = True, _maxCorners=10000, _qualityLevel=0.005, _minDistance=10):

	corner_points = []

	for index, image in enumerate(images):
		corner_point_vec = np.int0(cv2.goodFeaturesToTrack(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), _maxCorners, _qualityLevel, _minDistance))
		corner_points.append(corner_point_vec)

		if plot:
			plot_corners(image, corner_point_vec, images_names[index], suffix='corners')


	return corner_points



def ANMS(all_corners, images, images_names, N_best, plot=True):

	all_best_corners = []
	for index, image in enumerate(images):
		corners = all_corners[index]
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		N_s = corners.shape[0]

		r = [[sys.maxsize, [_, _]] for _ in range(N_s)]
		ED = 0
		for i in range(N_s):
			for j in range(N_s):
				x_i , y_i = corners[i][0][:]
				x_j, y_j = corners[j][0][:]
				if gray_image[y_i, x_i] > gray_image[y_j, x_j]:
					ED = (x_j - x_i) ** 2 + (y_j - y_i) ** 2
				if ED < r[i][0]:
					r[i][0] = ED
					r[i][1] = x_i, y_i
		r.sort()
		r.reverse()
		best_corners = np.delete(np.array(r[0:N_best]), 0, 1)
		all_best_corners.append(best_corners)
		if plot:
			plot_corners(image, best_corners, images_names[index], suffix='anms')

	return all_best_corners




def feature_descriptors(images, images_names, best_corners, patch_size, plot=True):

	filter_size = 3
	sampling_size = 5

	num_corners = best_corners[0].shape[0]
	all_features = []


	for index, image in enumerate(images):
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		paded_img = np.pad(gray_image, (int(patch_size/2)), 'constant', constant_values=0)
		features = []
		for j, corners in enumerate(best_corners[index]):
			filtered = cv2.GaussianBlur(crop(paded_img, patch_size, corners[0]),(filter_size,filter_size),0)
			feature_vec = standardize(sampling(filtered, sampling_size).flatten())
			features.append(feature_vec)
			if plot:
				# cv2.imwrite('./features/'+ images_names[index] + '/feature_vector_' + str(j) + '.png',
				# 			np.reshape(feature_vec, (8,8))/np.max(np.reshape(feature_vec, (8,8))))
				plt.imshow(np.reshape(feature_vec, (8, 8)) / np.max(np.reshape(feature_vec, (8, 8))), cmap='gray')
				plt.savefig('./features/'+ images_names[index] + '/feature_vector_' + str(j) + '.png')
		all_features.append(np.array(features))
	return all_features







def crop(paded_image, patch_size, corner_point):
	start_x = corner_point[0]
	start_y = corner_point[1]
	return paded_image[start_y: start_y + patch_size, start_x: start_x + patch_size]

def standardize(vec):
	vec = vec - np.mean(vec)
	vec = vec / np.std(vec)
	return vec

def sampling(array, sampling_size):
	return array[0::sampling_size, 0::sampling_size]


def feature_matching(images, images_names, best_corners, all_features, plot=True):

	num_points = best_corners[0].shape[0]
	per = permutations(images.__len__())
	all_matches = []
	for index, pair in enumerate(per):
		matches, match_flag = find_match(
			best_corners[pair[0]], best_corners[pair[1]], all_features[pair[0]], all_features[pair[1]], threshold=0.5)
		all_matches.append(np.array(matches))
		if plot:
			match_featues_plot(images, images_names, pair, matches)
	return all_matches, match_flag





def distance(vec1, vec2):
	diff = vec1 - vec2
	return (diff ** 2).sum()

def find_match(points1, points2, features1, features2, threshold):
	match_flag = True
	num_points_image1 = features1.shape[0]
	num_points_image2 = features2.shape[0]
	match = []
	for i in range (num_points_image1):
		score_maps = []
		for j in range (num_points_image2):
			score_maps.append([distance(features1[i], features2[j]), [points1[i], points2[j]]])
		score_maps.sort()
		if float(score_maps[0][0]) / float(score_maps[1][0]) < threshold:
			match.append(score_maps[0][1])
	if match.__len__() < 30:
		match_flag = False
	return np.array(match), match_flag

def permutations(num_elements):
	if num_elements == 2:
		return np.array([[0, 1]])
	else:
		return np.array([[0, 1], [1, 2]])


def hconcat(im1, im2):
	result = np.zeros((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], im1.shape[2]), type(im1.flat[0]))
	result[0:im1.shape[0], 0:im1.shape[1]] = im1
	result[0:im2.shape[0], im1.shape[1]:im1.shape[1] + im2.shape[1]] = im2

	return result



def match_featues_plot(images, images_names, pair, matches, suffix='matching'):
	pic = hconcat(images[pair[0]], images[pair[1]])
	shift = images[pair[0]].shape[1]

	for index, match in enumerate(matches):
		cv2.line(pic, match[0][0], (match[1][0][0] + shift, match[1][0][1]), (0, 255, 255), 1)
		cv2.circle(pic, match[0][0], 3, (255, 0, 0), -1)
		cv2.circle(pic, (match[1][0][0] + shift, match[1][0][1]), 3, (0, 0, 255), -1)

	cv2.imwrite(images_names[pair[0]] + '_' + images_names[pair[1]] + '_' + suffix + '.png', pic)



def rasnac(images, images_names, all_matches, N_max=3000, percentage=0.9, threshold=50.0, plot=True):
	all_rasnac_matches = []
	all_HGs = []
	per = permutations(images.__len__())
	for index, pair in enumerate(per):
		rasnac_matches, estimated_HG = rasnac_computation(images, images_names, all_matches, index, N_max=N_max, percentage=percentage, threshold=threshold)
		all_rasnac_matches.append(rasnac_matches)
		all_HGs.append(estimated_HG)
		if plot:
			match_featues_plot(images, images_names, pair, rasnac_matches, suffix='rasnac')

	return all_rasnac_matches, all_HGs




def rasnac_computation(images, images_names, all_matches, index, N_max=3000, percentage=0.9, threshold=50.0, plot=True):

	num_matches = all_matches[index].shape[0]
	max_inliers = 0

	for _ in range(N_max):
		desired_index = []
		rand_index = [np.random.randint(0, num_matches) for _ in range(4)]
		p, p_ = get_matches(rand_index, index, all_matches)
		H = cv2.getPerspectiveTransform(p.astype(np.float32), p_.astype(np.float32))

		num_inliers = 0
		for ind, match in enumerate(all_matches[index]):


			predict = np.matmul(H, np.array([match[0][0][0], match[0][0][1], 1]))

			if predict[2] == 0: predict[2] = 0.000001

			predict = np.array([predict[0]/predict[2], predict[1]/predict[2]]).astype(np.float32)

			if np.sqrt(distance(match[1][0], predict)) < threshold:
				num_inliers += 1
				desired_index.append(ind)

		pts1 = []
		pts2 = []
		if max_inliers < num_inliers:
			max_inliers = num_inliers
			[pts1.append([all_matches[index][i][0][0]]) for i in desired_index]
			[pts2.append([all_matches[index][i][1][0]]) for i in desired_index]
			H_, status = cv2.findHomography(np.float32(pts1), np.float32(pts2))
			if num_inliers > percentage * num_matches:
				break

	return np.array([all_matches[index][i] for i in desired_index]), np.array(H_)






def get_matches(rand_index, index, all_matches):
	p = [all_matches[index][rand_index[i]][0][0] for i in range(rand_index.__len__())]
	p_ = [all_matches[index][rand_index[i]][1][0] for i in range(rand_index.__len__())]

	return np.array(p), np.array(p_)





def warping_blending(_images, _images_names, plot=True):
	left_image = _images[0]
	left_name = _images_names[0]
	for i, right_image in enumerate(_images[1:]):
		images = [left_image, right_image]
		right_name = _images_names[i+1]
		images_names = [left_name, right_name]


		corners = corner_detection(images, images_names, plot=True, _maxCorners=1000000, _qualityLevel=0.05,
								   _minDistance=5)
		best_corners = ANMS(corners, images, images_names, N_best=1000, plot=True)
		all_features = feature_descriptors(images, images_names, best_corners, patch_size=40, plot=False)
		all_matches, match_flag = feature_matching(images, images_names, best_corners, all_features, plot=plot)
		if not match_flag:
			continue
		all_rasnac_matches, HG = rasnac(images, images_names, all_matches, N_max=50000, percentage=0.9,
											 threshold=100.0, plot=plot)

		result = warp_images(right_image, left_image, HG[0])
		left_image = result
		left_name = left_name + right_name

	if plot:
		cv2.imwrite(left_name + right_name + '_pano.png', result)
	return result




def warp_images(image1, image2, H):

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    p1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    p2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(p2, H)
    pts = np.concatenate((p1, pts2_), axis=0)
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    out_image = cv2.warpPerspective(image2, H_t.dot(H), (x_max-x_min, y_max-y_min))
    out_image[t[1]:h1+t[1], t[0]:w1+t[0]] = image1
    return out_image


def main():

	"""
	
	Read a set of images for Panorama stitching
	"""

	set_names = ['Set1' , 'Set2', 'Set3', 'TestSet1', 'TestSet2', 'TestSet3', 'TestSet4']
	for set_name in set_names:
		images = []
		images_names = []
		os.chdir("..")
		cwd = os.getcwd() + '/Data/Train/' + set_name
		os.chdir(os.getcwd() + '/Code')
		for root, dirs, files in os.walk((cwd)):
			for image in files:
				images.append(cv2.imread(root + '/' + image))
				images_names.append(set_name + '_' + image.split('.')[0])

		"""
		Corner Detection
		Save Corner detection output as corners.png
		"""
		corner_points = corner_detection(images, images_names)

		corners = corner_detection(images, images_names, plot=True, _maxCorners=1000000, _qualityLevel=0.02,
								   _minDistance=8)


		"""
		Perform ANMS: Adaptive Non-Maximal Suppression
		Save ANMS output as anms.png
		"""

		best_corners = ANMS(corners, images, images_names, N_best=1000, plot=True)



		"""
		Feature Descriptors
		Save Feature Descriptor output as FD.png
		"""

		all_features = feature_descriptors(images, images_names, best_corners, patch_size=40, plot=True)



		"""
		Feature Matching
		Save Feature Matching output as matching.png
		"""
		all_matches, match_flag = feature_matching(images, images_names, best_corners, all_features, plot=True)


		"""
		Refine: RANSAC, Estimate Homography
		"""
		all_rasnac_matches , all_HGs = rasnac(images, images_names, all_matches, N_max=5000, percentage=0.9, threshold=100.0, plot=True)



		"""
		Image Warping + Blending
		Save Panorama output as mypano.png
		"""
		warping_blending(images, images_names, plot=False)


    
if __name__ == '__main__':
    main()
 



