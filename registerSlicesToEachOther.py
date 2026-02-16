#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 03:25:30 2026

@author: zjpeters
"""
import os
import pandas as pd
import h5py
# from scipy.sparse import csr_matrix
import numpy as np
from matplotlib import pyplot as plt
from brainglobe_atlasapi import BrainGlobeAtlas
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import scipy.sparse as sp_sparse
import scipy.spatial as sp_spatial
import random
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import nibabel as nib
from scipy import ndimage
import ants
from skimage.transform import rescale, rotate

fileLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')
# list generated from selectGenePatterns, selecing only genes with strong patterns
listForGeneImage = ['Prdm12', 'Mal', 'Nts', 'Cbln1', 'Col1a1', 'Cdh13', 'Ramp1', 'Rgs6', 'Gpr88', 'Rorb', 'Slc17a7', 'Pou3f1', 'Zfpm2', 'Pvalb', 'Slc1a3']
sampleRotations = pd.read_csv(os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','rotationsPerSample.csv'))

# create function to display gene image from single slice
def displaySingleSliceGeneImage(sliceNumber, geneMatrix, geneList, micronsToDisplay=25, pixelCombination='additive', displayImage=True):
    sliceIdx = np.where(slice_codes == sliceNumber)[0]
    slice_coordinates = tissue_coordinates[sliceIdx, :]/micronsToDisplay
    image_size = np.round(np.max(tissue_coordinates, axis=0))/micronsToDisplay
    gene_image = np.zeros([int(image_size[0])+1, int(image_size[1])+1])
    for actGene in enumerate(geneList):
        geneArray = geneMatrix[sliceIdx,:].todense()[:, [actGene[0]]]
        geneArray = geneArray/np.max(geneArray)
        geneMask = np.squeeze(np.array(geneArray > 0))
        geneCoordinates = slice_coordinates[geneMask,:]
        genePixelNum = actGene[0] + 1
        for coordinate in enumerate(geneCoordinates):
            xCoor = int(coordinate[1][0])
            yCoor = int(coordinate[1][1])
            # print(coordinate[1], xCoor)
            if pixelCombination == 'additive':
                # print(xCoor, yCoor, genePixelNum)
                gene_image[xCoor, yCoor] += genePixelNum
            elif pixelCombination == 'replace':
                gene_image[xCoor, yCoor] = genePixelNum
            elif pixelCombination == 'geneExpressionScaledReplace':
                gene_image[xCoor, yCoor] = genePixelNum * geneArray[:,coordinate[0]]
            elif pixelCombination == 'geneExpressionScaledAdditive':
                gene_image[xCoor, yCoor] += genePixelNum * geneArray[coordinate[0]]
            elif pixelCombination == 'geneExpression':
                gene_image[xCoor, yCoor] += geneArray[coordinate[0]]
    gene_image = np.array(gene_image / np.max(gene_image) * 255, dtype='uint8')
    if displayImage == True:
        plt.figure()
        plt.imshow(gene_image, cmap='gray')    
        # plt.scatter(slice_coordinates[:,0], slice_coordinates[:,1], s=1)
        plt.show()
    return gene_image

#%% load data from H5 files

f = h5py.File(fileLocation)
x_attrs = dict(f['X'].attrs)
gene_matrix = sp_sparse.csr_array((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
gene_list = np.array(f['var']['_index'])
tissue_coordinates = f['obsm']['spatial'][:,0:2]
# could potentially add slice_codes as the 3rd column in tissue coordinates
slice_codes = (f['obs']['brain_section_barcode']['codes'][:])
ccfCoordinates = np.array([f['obs']['x_CCF'][:], f['obs']['y_CCF'][:], f['obs']['z_CCF'][:]]).T
f.close()

# convert gene list from bytes to strings

for i in range(len(gene_list)):
    gene_list[i] = gene_list[i].decode("utf-8")

gene_list = list(gene_list)

#%% get approximate z-slice in CCF for each slice
"""
taking the CCF coordinates as a starter, will use the average slice location and 
then sort those numbers for an estimate of the slice order
""" 
sliceMean = []
for sliceNumber in range(58):
    sliceIdx = np.where(slice_codes == sliceNumber)[0]
    slice_coordinates = ccfCoordinates[sliceIdx, :]
    sliceMean.append(np.mean(slice_coordinates[:,0]))
sliceMean = np.array(sliceMean)
sliceSortIdx = np.argsort(sliceMean)

#%% reduce gene matrix to only genes of interest
gene_list_idx = []
for gene_name in listForGeneImage:
    gene_list_idx.append(gene_list.index(gene_name))
gene_list_idx = np.array(gene_list_idx)

short_gene_matrix = gene_matrix[:,gene_list_idx]

#%% loop through all samples and plot gene image
plt.close('all')
allImages  = {"sliceNumber":[], "image":[], "rotation":[], "rotatedImage":[]}
for i in sliceSortIdx:
    image = displaySingleSliceGeneImage(i, short_gene_matrix, listForGeneImage, pixelCombination='additive', displayImage=False)
    # plt.title(f'Slice {i}')
    # plt.show()
    allImages["sliceNumber"].append(i)
    allImages["image"].append(image)
    allImages["rotation"].append(sampleRotations["Rotation"][np.where(sampleRotations['SliceNumber'] == i)[0][0]])
    
#%% test on all samples using csv or rotations
plt.close('all')

for i in range(len(sampleRotations['SliceNumber'])):
    rotImage = rotate(allImages['image'][i], allImages['rotation'][i], resize=True)
    allImages['rotatedImage'].append(rotImage)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(allImages['image'][i])
    ax[1].imshow(rotImage)
    plt.title(allImages["sliceNumber"][i])
    plt.show()

#%% select images to test translation
"""
Slice 36 (idx 30 within the allImages dictionary) has a good orientation as base
"""
plt.close('all')
fixedImage = ants.from_numpy(allImages["rotatedImage"][30])
movingImage = ants.from_numpy(allImages["rotatedImage"][31])

#%% run rigid transform to see result
affXfm = ants.registration(fixed=fixedImage, moving=movingImage, type_of_transform='Rigid')
# grad_step=0.2, flow_sigma=1, total_sigma=0, aff_metric='mattes', 
#  aff_sampling=32, aff_random_sampling_rate=0.2, syn_metric='mattes',
#  syn_sampling=32, reg_iterations=[100,40,20,0], aff_iterations=[2100,1200,1200,10],
#  aff_shrink_factors=[6,4,2,1], aff_smoothing_sigmas=[3,2,1,0]
fig, ax = plt.subplots(1,2)
ax[0].imshow(fixedImage.numpy())
ax[1].imshow(affXfm['warpedmovout'].numpy())
plt.show()