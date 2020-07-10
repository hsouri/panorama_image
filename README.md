# Panorama Image

![Repo List](figures/over.jpg)

# Introcduction

The purpose of this project is to stitch two or more images in order to create one seamless panorama image. Each image should have few repeated local features (∼ 30-50% or more, emperically chosen). The following method of stitching images should work for most image sets.

![Repo List](figures/head.png)

# Data Set

You need to capture two sets of images in order to stitch a seamless panorama. Each sequence should have at-least 3 images with ∼ 30-50 % image overlap between them. Feel free to check the sample images given to you in Data\Train\ folder.


# Classical Approach

An overview of the panorama stitching using the traditional approach is given below.

![Repo List](figures/over.png)

## Corner Detection

The first step in stitching a panorama is extracting corners like most computer vision tasks. Here we will use either Harris corners or Shi-Tomasi corners. We use cv2.cornerHarris or cv2.goodFeaturesToTrack to implement this part.

## Adaptive Non-Maximal Suppression (ANMS)
The objective of this step is to detect corners such that they are equally distributed across the image in order to avoid weird artifacts in warping.

In a real image, a corner is never perfectly sharp, each corner might get a lot of hits out of the N strong corners - we want to choose only the Nbest best corners after ANMS. In essence, we get a lot more corners than we should! ANMS will try to find corners which are true local maxima. The algorithm for implementing ANMS is given below.
![Repo List](figures/ANMS.png)


Output of corner detection and ANMS is shown below. Observe that the output of ANMS is evenly distributed strong corners.

![Repo List](figures/ANMSout.png)

## Feature Descriptor

In the previous step, we found the feature points (locations of the Nbest best corners after ANMS are called the feature point locations). we now need to describe each feature point by a feature vector, this is like encoding the information at each feature point by a vector. One of the easiest feature descriptor is described next.

We take a patch of size 40×40 centered (this is very important) around the keypoint/feature point. Then apply gaussian blur. Sub-sample the blurred output (this reduces the dimension) to 8×8. Then reshape to obtain a 64×1 vector. Then we standardize the vector to have zero mean and variance of 1. Standardization is used to remove bias and to achieve some amount of illumination invariance.
