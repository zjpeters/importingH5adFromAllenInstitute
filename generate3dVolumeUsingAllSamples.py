#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 07:19:47 2026

@author: zjpeters
"""

import os
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
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


fileLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')
# list generated from selectGenePatterns, selecing only genes with strong patterns
listForGeneImage = ['Prdm12', 'Mal', 'Nts', 'Cbln1', 'Col1a1', 'Cdh13', 'Ramp1', 'Rgs6', 'Gpr88', 'Rorb', 'Slc17a7', 'Pou3f1', 'Zfpm2', 'Pvalb', 'Slc1a3']

#%% load all slices into 3d volume

f = h5py.File(fileLocation)
x_attrs = dict(f['X'].attrs)
gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
gene_list = np.array(f['var']['_index'])
# tissue_coordinates = f['obsm']['spatial'][:,0:2]
slice_codes = (f['obs']['brain_section_barcode']['codes'][:])
ccfCoordinates = np.array([f['obs']['x_CCF'][:], f['obs']['y_CCF'][:], f['obs']['z_CCF'][:]]).T
f.close()

# convert gene list from bytes to strings

for i in range(len(gene_list)):
    gene_list[i] = gene_list[i].decode("utf-8")

gene_list = list(gene_list)
#%% reduce gene matrix to only genes of interest
gene_list_idx = []
for gene_name in listForGeneImage:
    gene_list_idx.append(gene_list.index(gene_name))
gene_list_idx = np.array(gene_list_idx)

short_gene_matrix = gene_matrix[:,gene_list_idx]

#%% 
roundedCoordinates = np.round(ccfCoordinates)
del(gene_matrix, ccfCoordinates)
# find unique coordinates
roundedCoordinates = roundedCoordinates
uniqueCoordinates = np.unique(roundedCoordinates, axis=0)


condensedGeneMatrixSavePath = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'condensedGeneMatrixFor3dVolume.npz')
    # check whether similarity matrix has already been calculated
#%%
if os.path.exists(condensedGeneMatrixSavePath):
    condensedGeneMatrix = sp_sparse.load_npz(condensedGeneMatrixSavePath)
else:
    # create empty gene matrix to assign new valuess to
    n = short_gene_matrix.shape[0]
    nInc = n/20
    percentLocations = np.arange(0, n, nInc, dtype='int32')
    percents = np.arange(5,101, 5)
    condensedGeneMatrix = np.zeros([uniqueCoordinates.shape[0], short_gene_matrix.shape[1]])
    for coor1 in range(uniqueCoordinates.shape[0]):
        uniqueIdx = np.where(np.all(roundedCoordinates == uniqueCoordinates[coor1,:], axis=1))[0]
        # leaving it as a mean for both multiple and singleton indices, since it won't effect singleton
        condensedGeneMatrix[coor1,:] = np.mean(short_gene_matrix[uniqueIdx,:], axis=0)
        # print(uniqueIdx)
        if coor1 in percentLocations:
            print(f"Calculation {percents[list(percentLocations).index(coor1)]}% completed")
    condensedGeneMatrix = sp_sparse.csr_array(condensedGeneMatrix)
    sp_sparse.save_npz(condensedGeneMatrixSavePath, condensedGeneMatrix)
#%% create empty array to create 3d volume
# size of 25um allen atlas [528, 320, 456]
"""
49 cells have some coordinate below zero, must think about whether to drop
"""

maskZeroCoorsIdx = np.any(uniqueCoordinates < 0, axis=1)
maskZeroCoors = uniqueCoordinates[maskZeroCoorsIdx,:]
maskCondensedMatrix = condensedGeneMatrix[maskZeroCoorsIdx,:]
volArray = np.zeros([528, 350, 480])
pixelCombination = 'additive'
for actGene in enumerate(listForGeneImage):
    geneArray = condensedGeneMatrix.todense()[:, actGene[0]]
    geneArray = geneArray/np.max(geneArray)
    geneMask = np.squeeze(np.array(geneArray > 0))
    geneCoordinates = uniqueCoordinates[geneMask,:]
    genePixelNum = actGene[0] + 1
    print(genePixelNum)
    for coordinate in enumerate(geneCoordinates):
        xCoor = int(geneCoordinates[coordinate[0],0])
        yCoor = int(geneCoordinates[coordinate[0],1])
        zCoor =  int(geneCoordinates[coordinate[0],2])
        if pixelCombination == 'additive':
            volArray[xCoor, yCoor, zCoor] += genePixelNum
        elif pixelCombination == 'replace':
            volArray[xCoor, yCoor, zCoor] = genePixelNum
        elif pixelCombination == 'geneExpressionScaledReplace':
            volArray[xCoor, yCoor, zCoor] = genePixelNum * geneArray[:,coordinate[0]]
        elif pixelCombination == 'geneExpressionScaledAdditive':
            volArray[xCoor, yCoor, zCoor] += genePixelNum * geneArray[:,coordinate[0]]
geneImage = np.array(volArray / np.max(volArray) * 255, dtype='uint8')

#%% convert to nifti

affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.025, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])
niiImage = nib.Nifti1Image(geneImage, affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'geneAdditiveNii.nii'))

#%% create binary image from coordinates

volArray = np.zeros([528, 360, 480])
for coor in uniqueCoordinates.T:
    volArray[int(coor[1]), int(coor[2]), int(coor[0])] = 1
niiImage = nib.Nifti1Image(volArray, affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'binNii.nii'))

#%% use mean gene expression
volArray = np.zeros([528, 350, 480])
geneArray = np.mean(condensedGeneMatrix, axis=1)
geneArray = geneArray/np.max(geneArray)
geneMask = np.squeeze(np.array(geneArray > 0))
geneCoordinates = uniqueCoordinates[geneMask,:]
geneArrayMasked = geneArray[geneMask,:]
for coordinate in enumerate(geneCoordinates):
    xCoor = int(geneCoordinates[coordinate[0],0])
    yCoor = int(geneCoordinates[coordinate[0],1])
    zCoor =  int(geneCoordinates[coordinate[0],2])
    volArray[xCoor, yCoor, zCoor] += geneArrayMasked[coordinate[0]]
geneImage = np.array(volArray / np.max(volArray) * 255, dtype='uint8')

niiImage = nib.Nifti1Image(geneImage, affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'geneMeanNii.nii'))

#%% use total gene expression
volArray = np.zeros([528, 350, 480])
geneArray = np.sum(condensedGeneMatrix, axis=1)
geneArray = geneArray/np.max(geneArray)
geneMask = np.squeeze(np.array(geneArray > 0))
geneCoordinates = uniqueCoordinates[geneMask,:]
geneArrayMasked = geneArray[geneMask,:]
for coordinate in enumerate(geneCoordinates):
    xCoor = int(geneCoordinates[coordinate[0],0])
    yCoor = int(geneCoordinates[coordinate[0],1])
    zCoor =  int(geneCoordinates[coordinate[0],2])
    volArray[xCoor, yCoor, zCoor] += geneArrayMasked[coordinate[0]]
geneImage = np.array(volArray / np.max(volArray) * 255, dtype='uint8')

niiImage = nib.Nifti1Image(geneImage, affMatrix)

nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'geneSumNii.nii'))

#%% try to interpolate empty space between cells

cellDistances = ndimage.distance_transform_edt(geneImage)
#%% display distances beside gene image

sliceToDisplay = 157
plt.close('all')
fig, ax = plt.subplots(1,2)
ax[0].imshow(geneImage[sliceToDisplay,:,:])
ax[1].imshow(cellDistances[sliceToDisplay,:,:])
plt.show()

#%% load allen template from BrainGlobe

# allen CCF for merscope data is aligned to 25um
atlas='allen_mouse_25um'
ara_data = BrainGlobeAtlas(atlas, check_latest=False)

#%% display template with gene image
templateSlice = ara_data.reference[sliceToDisplay,:,:]
plt.close('all')
plt.imshow(templateSlice)
plt.imshow(geneImage[sliceToDisplay,:,:], alpha=0.4)
plt.show()

#%% display grid of multiple template slices with the corresponding gene image
# first slice in gene image is ~157, last ~470
nOfGrids = 49
nRows = 7
nCols = 7
# fig,ax = plt.subplots(len(cellTypes), len(sparseClusters), figsize=(21, 11))
plt.close('all')
fig,ax = plt.subplots(nRows, nCols)
sliceArray = np.linspace(157, 470, nOfGrids, dtype='int32')
i = 0
j = 0
for sliceToDisplay in enumerate(sliceArray):
    print(sliceToDisplay)
    templateSlice = ara_data.reference[sliceToDisplay[1],:,:]
    ax[i,j].imshow(templateSlice, cmap='gray')
    ax[i,j].imshow(geneImage[sliceToDisplay[1],:,:], alpha=0.4)
    ax[i,j].tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False)
    ax[i,j].set_title(f'Slice {sliceToDisplay[1]}')
    if i < nCols-1:
        i += 1
    else:
        i = 0
        j += 1
plt.show()

#%% update gene pattern selection to randomly select cells to use for calculation
def measureTranscriptomicSimilarity(geneMatrix, edgeList='FullyConnected', measurement='cosine', axis=1, nRandomCells=5000):
    """
    measure the transcriptomic similarity/distance between two spots
    ----------
    Parameters
    ----------
    geneMatrix: float array
        2D matrix of genetic or transcriptomic data, organized [gene,spot]
    edgeList: Nx2 int list
        List of edges to be measured for distance, [[spot1,spot2],[spot1,spot3],...]
    measurement: str
        Choice of measurement metric, default='cosine'
        'cosine' - cosine similarity
        'pearson' - Pearson's R correlation (**not yet implemented**)
    axis: int
        Which axis to run the similarity metric on, default=1, for cells/spots
    """
    
    # dataSimMatrix = []   
    n = geneMatrix.shape[axis]
    nInc = n/20
    percentLocations = np.arange(0, n, nInc, dtype='int32')
    percents = np.arange(5,101, 5)
    if axis == 0:
        geneMatrix = geneMatrix.tocsr()
    if edgeList == 'FullyConnected':
        dataSimMatrix = sp_sparse.lil_matrix((geneMatrix.shape[axis],geneMatrix.shape[axis]))
        for i in range(geneMatrix.shape[axis]):
            for j in range(i, geneMatrix.shape[axis]):
                if axis==1:
                    I = np.ravel(geneMatrix[:,i].todense())
                    J = np.ravel(geneMatrix[:,j].todense())
                elif axis==0:
                    I = np.ravel(geneMatrix[i,:].todense())
                    J = np.ravel(geneMatrix[j,:].todense())
                cs = sp_spatial.distance.cosine(I,J)
                dataSimMatrix[i,j] = float(cs)
                dataSimMatrix[j,i] = float(cs)
            if i in percentLocations:
                print(f"Calculation {percents[list(percentLocations).index(i)]}% completed")
        dataSimMatrix = dataSimMatrix.tocsc()
    if edgeList == 'RandomCells':
        randomCells = random.sample(range(0, geneMatrix.shape[0]), nRandomCells)
        geneMatrix = geneMatrix[randomCells,:]
        dataSimMatrix = sp_sparse.lil_matrix((geneMatrix.shape[axis],geneMatrix.shape[axis]))
        for i in range(geneMatrix.shape[axis]):
            for j in range(i, geneMatrix.shape[axis]):
                if axis==1:
                    I = np.ravel(geneMatrix[:,i].todense())
                    J = np.ravel(geneMatrix[:,j].todense())
                elif axis==0:
                    I = np.ravel(geneMatrix[i,:].todense())
                    J = np.ravel(geneMatrix[j,:].todense())
                cs = sp_spatial.distance.cosine(I,J)
                dataSimMatrix[i,j] = float(cs)
                dataSimMatrix[j,i] = float(cs)
            if i in percentLocations:
                print(f"Calculation {percents[list(percentLocations).index(i)]}% completed")
        dataSimMatrix = dataSimMatrix.tocsc()
    return dataSimMatrix

#%%
def selectGenePatterns(processedSample, k=16, restrictByVariance=False, nRandomCells=0):
    if 'geneMatrixLog2' in processedSample:
        geneMatrix = processedSample['geneMatrixLog2']
    elif 'geneMatrixMasked' in processedSample:
        geneMatrix = processedSample['geneMatrixMasked']
    else:
        geneMatrix = processedSample['geneMatrix']
    if 'geneList' in processedSample:
        listToSearch = processedSample['geneList']
    elif 'geneListMasked' in processedSample:
        listToSearch = processedSample['geneListMasked']
        
    if restrictByVariance==True:
        geneMatrixSquared = geneMatrix.multiply(geneMatrix)
        geneMatrixMean = geneMatrix.mean(axis=1)
        geneMatrixVariance = np.squeeze(np.array(geneMatrixSquared.mean(axis=1) - np.square(geneMatrixMean)))
        geneVarianceSortIdx = np.argsort(geneMatrixVariance)
        nOfTopVariance = 501
        topVarianceIdx = np.squeeze(geneVarianceSortIdx.T[-nOfTopVariance:-1])
        geneMatrix = geneMatrix[topVarianceIdx, :]
        listToSearch = np.array(listToSearch)[topVarianceIdx]
    # test cosine similarity of genes, i.e. how similar each gene is to each other
    if nRandomCells > 0:
        similarityMatrix = measureTranscriptomicSimilarity(geneMatrix, axis=1, edgeList='RandomCells', nRandomCells=nRandomCells)
    else:
        similarityMatrix = measureTranscriptomicSimilarity(geneMatrix, axis=0)
    # look at eigenvalues for gene clustering
    Wcontrol = (similarityMatrix.todense() - np.nanmin(similarityMatrix.todense()))/(np.nanmax(similarityMatrix.todense()) - np.nanmin(similarityMatrix.todense()))
    Wcontrol[Wcontrol==1] = 0
    # Wcontrol[np.isnan(Wcontrol)] = 0
    Dcontrol = np.diag(sum(Wcontrol))
    Lcontrol = Dcontrol - Wcontrol
    eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=k)
    eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
    eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])
    
    clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))
    sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:k]), cluster_labels)

    # loop through the clusters and find the genes with the highest silhouette value in each cluster
    geneListForImage = []
    geneCellCount = []
    for i in range(k):
        clusterIdx = np.where(cluster_labels == i)
        silhouettePerCluster = sample_silhouette_values[clusterIdx]
        topSilhouetteIdx = clusterIdx[0][np.argmax(silhouettePerCluster)]
        geneListForImage.append(listToSearch[topSilhouetteIdx])
        geneCellCount.append(np.sum(geneMatrix[topSilhouetteIdx,:].todense() > 0))
    geneCellCount = np.array(geneCellCount, dtype='int')
    # sort genes from gene expressed in fewest cells to gene expressed in most cells
    geneCellCountSortedIdx = np.argsort(geneCellCount)
    geneListSorted = []
    for i in geneCellCountSortedIdx[::-1]:
        print(geneListForImage[i], geneCellCount[i])
        geneListSorted.append(geneListForImage[i])
    return geneListSorted



