from __future__ import division

import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage import data, img_as_float, exposure, color, restoration
from skimage.morphology import erosion, binary_erosion, dilation
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from skimage.util import img_as_ubyte

from scipy.signal import convolve2d
import scipy.ndimage

import numpy as np
import pandas as pd
import sys
import copy

import os
import shutil


def preprocess_dataset(base_dir):
    covid_df = pd.DataFrame(columns=['image_id', 'is_covid'])
    non_covid_df = pd.DataFrame(columns=['image_id', 'is_covid'])

    for file in os.listdir(os.path.join(base_dir, 'CT_COVID')):
        new_row = pd.DataFrame(data={'image_id': [file], 'is_covid':[1]})
        covid_df = pd.concat([covid_df, new_row], ignore_index=True, sort=False)
        shutil.copy(base_dir + 'CT_COVID/' + file, base_dir + 'training_data/' + file)
        
    for file in os.listdir(os.path.join(base_dir, 'CT_NonCOVID')):
        new_row = pd.DataFrame(data={'image_id': [file], 'is_covid':[0]})
        non_covid_df = pd.concat([non_covid_df, new_row], ignore_index=True, sort=False)
        shutil.copy(base_dir + 'CT_NonCOVID/' + file, base_dir + 'training_data/' + file)
        
    df = pd.concat([covid_df, non_covid_df], ignore_index=True, sort=False)

    return df


def segment_lungs(image: np.ndarray, display=False)->np.ndarray:
    row_size= image.shape[0]
    col_size = image.shape[1]
    
    mean = np.mean(image)
    std = np.std(image)
    image = (image-mean)/std
    
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = image[int(col_size/5):int(col_size/5*4), int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max_ = np.max(image)
    min_ = np.min(image)
    
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    image[image==max_] = mean
    image[image==min_] = mean
    
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers) 
    thresh_image = np.where(image<threshold, 1.0, 0.0)  

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded_image = morphology.erosion(thresh_image,np.ones([2, 2]))
    dilated_image = morphology.dilation(eroded_image,np.ones([8, 8]))

    labels = measure.label(dilated_image, connectivity=2) 
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    
    for prop in regions:
        B = prop.bbox

        if (B[2]-B[0]<=row_size*1.0 and 
           B[2]-B[0]>=row_size*0.4 and 
           B[3]-B[1]<=col_size*0.8 and 
           B[3]-B[1]>=col_size * 0.2 and
           B[0]>=row_size*0.00 and 
           B[2]<=row_size*1.0):
            good_labels.append(prop.label)
            
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N, 1, 0)
    mask = morphology.dilation(mask,np.ones([18, 18])) 
    
    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(image, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_image, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilated_image, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*image, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
        
    return mask*images


def clean_noise(image: np.ndarray)->np.ndarray:
    image = np.where(image<-1, -1, image)
    image = np.where(image>1, 1, image)

    p2, p98 = np.percentile(image, (2, 98))
    image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    # Adaptive Equalization
    image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    
    psf = np.ones((5, 5)) / 25
    img = convolve2d(image_adapteq, psf, 'same')
    img += 0.1*img.std() * np.random.standard_normal(img.shape)
    
    deconvolved_image = restoration.wiener(img, psf, 1100)
    
    return deconvolved_image


def extract_glcm_features(image: np.ndarray)->np.ndarray:
    
    
    def mu(glcm: np.ndarray)->np.ndarray:
        m = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                m += i*glcm[i, j]
                
        return m


    def sigma_sq(glcm: np.ndarray)->np.ndarray:
        s = 0
        m = mu(glcm)

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                s += glcm[i, j]*((i - m)**2)
                
        return s


    def A(glcm: np.ndarray, correlation: np.ndarray)->np.ndarray:
        a = 0
        m = mu(glcm)
        s = sigma_sq(glcm)
        denom = np.sqrt(s)**3 * (np.sqrt(2*(1+correlation)))**3

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                nom = ((i+j-2*m)**3) * glcm[i, j]
                a += nom/denom
                
        return a


    def B(glcm: np.ndarray, correlation: np.ndarray)->np.ndarray:
        b = 0
        m = mu(glcm)
        s = sigma_sq(glcm)
        denom = 4*np.sqrt(s)**4 * ((1+correlation)**2)

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                nom = ((i+j-2*m)**4) * glcm[i, j]
                b += nom/denom
                
        return b


    def entropy(glcm: np.ndarray)-> np.ndarray:
        entropy = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                if glcm[i, j] != 0:
                    entropy += -np.log(glcm[i,j])*glcm[i, j]
                    
        return entropy


    def cluster_shade(glcm: np.ndarray, correlation: np.ndarray)->np.ndarray:
        
        return np.sign(A(glcm, correlation)) * np.absolute(A(glcm, correlation))**(1/3)


    def cluster_prominence(glcm: np.ndarray, correlation: np.ndarray)->np.ndarray:
        
        return np.sign(B(glcm, correlation)) * np.absolute(B(glcm, correlation))**(1/4)


    def difference_average(glcm: np.ndarray)->np.ndarray:
        da = 0
        for k in range(glcm.shape[0]-1):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if np.abs(i-j) == k:
                        da += k*glcm[i, k] 
                        
        return da    


    def difference_variance(glcm: np.ndarray)->np.ndarray:
        dv = 0
        m = mu(glcm)
        for k in range(glcm.shape[0]-1):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if np.abs(i-j) == k:
                        dv += (k-m)**2 * glcm[i, j]
                        
        return dv


    def difference_entropy(glcm: np.ndarray)->np.ndarray:
        de = 0
        for k in range(glcm.shape[0]-1):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if np.abs(i-j) == k and glcm[i, j] != 0:
                        de += -np.log(glcm[i, j]) * glcm[i, j]
                        
        return de


    def sum_average(glcm: np.ndarray)->np.ndarray:
        sa = 0
        for k in range(2, 2*glcm.shape[0]):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if i+j == k:
                        sa += k * glcm[i, j]
                        
        return sa


    def sum_variance(glcm: np.ndarray)->np.ndarray:
        sv = 0
        m = mu(glcm)
        for k in range(2, 2*glcm.shape[0]):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if i+j == k:
                        sv += (k-m)**2 * glcm[i, j]
                        
        return sv


    def sum_entropy(glcm: np.ndarray)->np.ndarray:
        se = 0
        for k in range(2, 2*glcm.shape[0]):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    if i+j == k and glcm[i, j] != 0:
                        se += np.log(glcm[i, j]) * glcm[i, j]
                        
        return -se


    def inverse_difference(glcm: np.ndarray)->np.ndarray:
        inv_d = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                inv_d += glcm[i, j] / (1+np.abs(i-j)) 
                
        return inv_d


    def normalized_inverse_difference(glcm: np.ndarray)->np.ndarray:
        ninv_d = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                ninv_d += glcm[i, j] / (1+np.abs(i-j)/glcm.shape[0]) 
                
        return ninv_d


    def inverse_difference_moment(glcm: np.ndarray)->np.ndarray:
        inv_m = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                inv_m += glcm[i, j] / (1+(i-j)**2)
                
        return inv_m


    def normalized_inverse_difference_moment(glcm: np.ndarray)->np.ndarray:
        ninv_m = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                ninv_m += glcm[i, j] / (1+(i-j)**2/glcm.shape[0]**2) 
                
        return ninv_m


    def inverse_variance(glcm: np.ndarray)->np.ndarray:
        inv_v = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                if j > i:
                    inv_v += glcm[i, j] / (i-j)**2
                    
        return 2 * inv_v


    def autocorrelation(glcm: np.ndarray)->np.ndarray:
        acr = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                acr += i*j*glcm[i, j]
                
        return acr


    def information_correlation_1(glcm: np.ndarray)->np.ndarray:
        HXY = 0
        HX = 0
        HXY1 = 0

        for i in range(glcm.shape[0]):
            if (np.sum(glcm[i]) !=0):
                HX -= np.sum(glcm[i]) * np.log(np.sum(glcm[i]))
            for j in range(glcm.shape[1]):
                if (glcm[i, j]!=0 and np.sum(glcm[i])!=0):
                    HXY -= glcm[i, j] * np.log(glcm[i, j])
                    HXY1 -= glcm[i, j] * np.log(np.sum(glcm[i])*np.sum(glcm[j]))

        return (HXY-HXY1) / HX


    def information_correlation_2(glcm: np.ndarray)->np.ndarray:
        HXY = 0
        HXY2 = 0

        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                if glcm[i, j]!=0 and np.sum(glcm[i])*np.sum(glcm[j])!=0:
                    HXY -= glcm[i, j]*np.log(glcm[i, j])
                    HXY2 -= np.sum(glcm[i]) * np.sum(glcm[j]) * np.log(np.sum(glcm[i])*np.sum(glcm[j]))

        return np.sqrt(1 - np.exp(2*(HXY2-HXY)))
    
    
    img_uint8 = img_as_ubyte(image)
    glcm = greycomatrix(img_uint8, distances=[1], angles=[1], levels=256,
                        symmetric=True, normed=True
                       )
    
    contrast_ = greycoprops(glcm, 'contrast')
    dissimilarity_ = greycoprops(glcm, 'dissimilarity')
    homogeneity_ = greycoprops(glcm, 'homogeneity')
    energy_ = greycoprops(glcm, 'energy')
    correlation_ = greycoprops(glcm, 'correlation')
    asm_ = greycoprops(glcm, 'ASM')
    entropy_ = entropy(glcm)

    cluster_shade_ = cluster_shade(glcm, correlation_)
    cluster_prominence_ = cluster_prominence(glcm, correlation_)

    max_prob_ = [np.max(glcm)]

    joint_average_ = mu(glcm)
    joint_variance_ = sigma_sq(glcm)

    difference_average_ = difference_average(glcm)
    difference_variance_ = difference_variance(glcm)
    difference_entropy_ = difference_entropy(glcm)

    sum_average_ = sum_average(glcm)
    sum_variance_ = sum_variance(glcm)
    sum_entropy_ = sum_entropy(glcm)

    inverse_difference_ = inverse_difference(glcm)
    normalized_inverse_difference_ = normalized_inverse_difference(glcm)

    inverse_difference_moment_ = inverse_difference_moment(glcm)
    normalized_inverse_difference_moment_ = normalized_inverse_difference_moment(glcm)

    inverse_variance_ = inverse_variance(glcm)

    autocorrelation_ = autocorrelation(glcm)

    information_correlation_1_ = information_correlation_1(glcm)
    information_correlation_2_ = information_correlation_2(glcm)

    feature_vector = {
                      "contrast": contrast_,
                      "dissimilarity": dissimilarity_,
                      "homogeneity": homogeneity_,
                      "energy": energy_,
                      "correlation": correlation_,
                      "asm": asm_,
                      "entropy": entropy_,
                      "cluster_shade": cluster_shade_,
                      "cluster_prominence": cluster_prominence_,
                      "max_prob": max_prob_,
                      "average": joint_average_,
                      "variance": joint_variance_,
                      "difference_average": difference_average_,
                      "difference_variance": difference_variance_,
                      "difference_entropy": difference_entropy_,
                      "sum_average": sum_average_,
                      "sum_variance": sum_variance_,
                      "sum_entropy": sum_entropy_,
                      "inverse_difference": inverse_difference_,
                      "normalized_inverse_difference": normalized_inverse_difference_,
                      "inverse_difference_moment": inverse_difference_moment_,
                      "normalized_inverse_difference_moment": normalized_inverse_difference_moment_,
                      "inverse_variance": inverse_variance_,
                      "autocorrelation": autocorrelation_,
                      "information_correlation_1": information_correlation_1_,
                      "information_correlation_2": information_correlation_2_
    }
    
    return feature_vector