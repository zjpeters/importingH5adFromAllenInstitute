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
sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
# sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import scipy.sparse as sp_sparse
import scipy.spatial as sp_spatial
import random
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import nibabel as nib
sys.path.insert(0, os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','code'))
# sys.path.insert(0,'/media/zjpeters/Expansion/merscopeDataFromAllenInstitute/code')
import allenMerscopeCode
import cv2
import nibabel as nib

# windows workstation locations
sourcedata = os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','sourcedata')
derivatives = os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','derivatives')
# linux locations
# sourcedata = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata')
# derivatives = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives')
h5adLocation = os.path.join(sourcedata,'mouse_638850_registered.h5ad')
# list generated from selectGenePatterns, selecing only genes with strong patterns
listForGeneImage = ['Prdm12', 'Mal', 'Nts', 'Cbln1', 'Col1a1', 'Cdh13', 'Ramp1', 'Rgs6', 'Gpr88', 'Rorb', 'Slc17a7', 'Pou3f1', 'Zfpm2', 'Pvalb', 'Slc1a3']

listOfFiles = ['mouse_609882_registered.h5ad', 'mouse_609889_registered.h5ad', 
               'mouse_638850_registered.h5ad', 'mouse_658801_registered.h5ad', 
               'mouse_687997_registered.h5ad', 'mouse_702265_registered.h5ad']

sample = allenMerscopeCode.loadSingleSliceFromH5ad(h5adLocation, 4, displayScatter=True)

#%% load dataset to get information about slice number, etc
"""
data stored in mouse_638850_registered.h5ad is stored differently than mouse_609882_registered_082725.h5ad

slice_codes are stored in: f['obs']['section']['codes'][:]
"""
# get the number of slices from the complete dataset
f = h5py.File(h5adLocation)
slice_codes = np.unique(f['obs']['section']['codes'][:])
f.close()
# load samples 1 by 1 and display gene images
plt.close('all')
for i in slice_codes:
    micron_size = 10
    sample = allenMerscopeCode.loadSingleSliceFromH5ad(h5adLocation, i)
    geneImage = allenMerscopeCode.displaySingleSliceGeneImage(sample, listForGeneImage, micronsToDisplay=micron_size, displayImage=False)
    print(np.mean(sample['ccfCoordinates'][:,0]))
    # cv2.imwrite(os.path.join(derivatives, f'gene_image_{micron_size}um_slice_{i}.png'), geneImage)
    # plt.close()

#%% create gene images in CCF space for single sample
plt.close('all')
# first two images are displaying all NaN for ccf coordinates, so skipping for now
meanCCFZ = []
allImagesCCF = []

for i in range(2,59):
    micron_size = 10
    sample = allenMerscopeCode.loadSingleSliceFromH5ad(h5adLocation, i)
    geneImage = allenMerscopeCode.displaySingleSliceGeneImageCCF(sample, listForGeneImage, displayImage=True)
    print(np.mean(sample['ccfCoordinates'][:,0]))
    meanCCFZ.append(np.mean(sample['ccfCoordinates'][:,0]))
    allImagesCCF.append(geneImage)

#%% create nifti from single sample 

affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.25, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])
niiImage = nib.Nifti1Image(np.array(allImagesCCF), affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'geneImage_mouse_638850_registered.nii'))

#%% create gene images in CCF space for all samples
plt.close('all')
# first two images are displaying all NaN for ccf coordinates, so skipping for now
meanAllSamplesCCFZ = []
allImagesAllSamplesCCF = []
for filename in listOfFiles:
    f = h5py.File(os.path.join(sourcedata, filename))
    slice_codes = np.unique(f['obs']['section']['codes'][:])
    f.close()
    nSlices = len(np.unique(slice_codes))
    for i in range(nSlices + 1):
        sample = allenMerscopeCode.loadSingleSliceFromH5ad(os.path.join(sourcedata, filename), i)
        if ~np.any(np.isnan(sample['ccfCoordinates'])) and len(sample['ccfCoordinates'] > 0):
            geneImage = allenMerscopeCode.displaySingleSliceGeneImageCCF(sample, listForGeneImage, displayImage=False, scaleImage=False)
            print(np.mean(sample['ccfCoordinates'][:,0]))
            meanAllSamplesCCFZ.append(np.mean(sample['ccfCoordinates'][:,0]))
            allImagesAllSamplesCCF.append(geneImage)
            
#%% sort images 
imageSortIdx = np.argsort(meanAllSamplesCCFZ)
sortedImageVolume = np.zeros([360,480,330])
for i in enumerate(imageSortIdx):
    print(i)
    sortedImageVolume[:,:,i[0]] = allImagesAllSamplesCCF[i[1]]

#%% create nifti file
affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.025, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])
niiImage = nib.Nifti1Image(np.array(sortedImageVolume), affMatrix)

nib.save(niiImage, os.path.join(derivatives, 'geneImage_mouse_all_samples_unscaled.nii'))

#%% loop through all samples from the allen institute and concatenate the ccf coordinates

allCoordinates = np.empty([0,3])
for filename in listOfFiles:
    print(filename)
    f = h5py.File(os.path.join(sourcedata, filename))
    sampleCoordinates = np.array([f['obs']['x_CCF'][:], f['obs']['y_CCF'][:], f['obs']['z_CCF'][:]]).T
    f.close()
    allCoordinates = np.append(allCoordinates, sampleCoordinates, axis=0)

roundedCoordinates = np.round(allCoordinates)
uniqueCoordinates = np.unique(roundedCoordinates, axis=0)

#%% create binary image from coordinates

affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.025, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])
volArray = np.zeros([528, 360, 480])
for coor in uniqueCoordinates:
    if np.all(coor > 0) and np.all(~np.isnan(coor)):
        volArray[int(coor[0]), int(coor[1]), int(coor[2])] = 1
        # volArray[int(coor[1]), int(coor[2]), int(coor[0])] = 1
niiImage = nib.Nifti1Image(volArray, affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'binNiiFromAllSamples.nii'))

#%% need to next add detailed gene information 
allCoordinates = np.empty([0,3])
for filename in listOfFiles:
    print(filename)
    f = h5py.File(os.path.join(sourcedata, filename))
    sampleCoordinates = np.array([f['obs']['x_CCF'][:], f['obs']['y_CCF'][:], f['obs']['z_CCF'][:]]).T
    f.close()
    allCoordinates = np.append(allCoordinates, sampleCoordinates, axis=0)

roundedCoordinates = np.round(allCoordinates)
uniqueCoordinates = np.unique(roundedCoordinates, axis=0)
