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

fileLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')

def loadSingleSliceFromH5ad(h5adLocation, sliceNumber, displayScatter=False):
    """
    

    Parameters
    ----------
    h5adLocation : TYPE
        Where the allen data is stored.
    sliceNumber : TYPE
        Overall slice number, not the slice from CCF.
    displayScatter : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    sample : TYPE
        DESCRIPTION.

    """
    f = h5py.File(h5adLocation)
    x_attrs = dict(f['X'].attrs)
    gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
    gene_list = np.array(f['var']['_index'])
    tissue_coordinates = f['obsm']['spatial'][:,0:2]
    slice_codes = (f['obs']['brain_section_barcode']['codes'][:])
    x_CCF = f['obs']['x_CCF'][:]
    y_CCF = f['obs']['y_CCF'][:]
    z_CCF = f['obs']['z_CCF'][:]
    f.close()
    sliceIdx = np.where(slice_codes == sliceNumber)[0]
    sample = {}
    sample['geneMatrix'] = gene_matrix[sliceIdx,:]
    sample['tissuePositionCoordinates'] = tissue_coordinates[sliceIdx,:]
    sample['geneList'] = list(gene_list)
    for i in range(len(sample['geneList'])):
        sample['geneList'][i] = sample['geneList'][i].decode("utf-8")
    sample['ccfCoordinates'] = np.array([x_CCF[sliceIdx], y_CCF[sliceIdx], z_CCF[sliceIdx]]).T
    if displayScatter == True:
        fig,ax = plt.subplots(1,1)
        ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1])
        ax.yaxis.set_inverted(True)
        plt.show()
        # plt.figure()
        # plt.scatter(sample['tissuePositionCoordinates'][:,0], sample['tissuePositionCoordinates'][:,1], s=3)
        # plt.show()
    return sample
# sample = loadSingleSliceFromH5ad(fileLocation, 18, displayScatter=True)

### editing from stanly codebase to account for allen data 
def createGeneImageFromProcessedSample(processedSample, listOfMarkerGenes, displayImage=True, pixelCombination='additive'):
    if 'tissueImageProcessed' in processedSample:
        geneImage = np.zeros_like(processedSample['tissueImageProcessed'])
    elif 'tissueImageRegistered' in processedSample:
        geneImage = np.zeros_like(processedSample['tissueImageRegistered'])
    if 'geneMatrixLog2' in processedSample:
        geneMatrix = processedSample['geneMatrixLog2']
    elif 'geneMatrixMasked' in processedSample:
        geneMatrix = processedSample['geneMatrixMasked']
    if 'geneList' in processedSample:
        listToSearch = processedSample['geneList']
    elif 'geneListMasked' in processedSample:
        listToSearch = processedSample['geneListMasked']
    if 'processedTissuePositionList' in processedSample:
        tissueCoordinates = processedSample['processedTissuePositionList']
    elif 'maskedTissuePositionList' in processedSample:
        tissueCoordinates = processedSample['maskedTissuePositionList']
    for actGene in enumerate(listOfMarkerGenes):
        geneIdx = listToSearch.index(actGene[1])
        geneArray = geneMatrix[geneIdx,:].todense()
        geneMask = np.squeeze(np.array(geneArray > 0))
        geneArray = geneArray[:,geneMask]
        geneArray = geneArray/np.max(geneArray)
        geneTissuePositions = tissueCoordinates[geneMask, :]
        genePixelNum = actGene[0] + 1
        for coordinate in enumerate(geneTissuePositions):
            xCoor = round(coordinate[1][1])
            yCoor = round(coordinate[1][0])
            if xCoor >= geneImage.shape[0]:
                xCoor = geneImage.shape[0] - 1
            if yCoor >= geneImage.shape[1]:
                yCoor = geneImage.shape[1] - 1
            if pixelCombination == 'additive':
                geneImage[xCoor, yCoor] += genePixelNum
            elif pixelCombination == 'replace':
                geneImage[xCoor, yCoor] = genePixelNum
            elif pixelCombination == 'binaryAdd':
                geneImage[xCoor, yCoor] += 1
            elif pixelCombination == 'geneExpressionScaledReplace':
                geneImage[xCoor, yCoor] = genePixelNum * geneArray[:,coordinate[0]]
            elif pixelCombination == 'geneExpressionScaledAdditive':
                geneImage[xCoor, yCoor] += genePixelNum * geneArray[:,coordinate[0]]
    geneImage = np.array(geneImage / np.max(geneImage) * 255, dtype='uint8')
    if displayImage==True:
        plt.figure()
        plt.imshow(geneImage, cmap='gray_r')
        plt.show()
    return geneImage

sample = loadSingleSliceFromH5ad(fileLocation, 18, displayScatter=True)
#%% load all slices into 3d volume

f = h5py.File(fileLocation)
x_attrs = dict(f['X'].attrs)
gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
gene_list = np.array(f['var']['_index'])
tissue_coordinates = f['obsm']['spatial'][:,0:2]
slice_codes = (f['obs']['brain_section_barcode']['codes'][:])
x_CCF = f['obs']['x_CCF'][:]
y_CCF = f['obs']['y_CCF'][:]
z_CCF = f['obs']['z_CCF'][:]
f.close()

sliceNumber = 50
sliceIdx = np.where(slice_codes == sliceNumber)[0]
sample = {}
sample['geneMatrix'] = gene_matrix[sliceIdx,:]
sample['tissuePositionCoordinates'] = tissue_coordinates[sliceIdx,:]
sample['geneList'] = list(gene_list)
for i in range(len(sample['geneList'])):
    sample['geneList'][i] = sample['geneList'][i].decode("utf-8")
sample['ccfCoordinates'] = np.array([x_CCF[sliceIdx], y_CCF[sliceIdx], z_CCF[sliceIdx]]).T

fig,ax = plt.subplots(1,1)
ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1])
ax.yaxis.set_inverted(True)
plt.show()

#%% display cells as 3d plot
### memory intensive

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_CCF, y_CCF, z_CCF)
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
dataSimMatrix = measureTranscriptomicSimilarity(gene_matrix, edgeList='RandomCells')

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
    eigvalControlSort = np.real(eigvalControl[eigvalControlSortIdx])
    
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

#%% test updated selectGenePatterns using allen data

topGeneList = selectGenePatterns(sample, k=25, nRandomCells=2000)
#%% display gene images
plt.close('all')
for i in topGeneList:
    geneIdx = sample['geneList'].index(i)
    fig,ax = plt.subplots(1,1)
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=np.squeeze(np.array(sample['geneMatrix'].todense()[:,geneIdx])), cmap='Reds', s=2)
    ax.yaxis.set_inverted(True)
    plt.title(i)
    plt.show()
#%% create gene image
listForGeneImage = ['Prdm12', 'Mal', 'Nts', 'Cbln1', 'Col1a1', 'Cdh13', 'Ramp1', 'Rgs6', 'Gpr88', 'Rorb', 'Slc17a7', 'Pou3f1', 'Zfpm2', 'Pvalb', 'Slc1a3']

"""
need to rework the create gene image to work with teh allen data that has no underlying image
"""
#%% create gene list from allen data
def createGeneImageFromAllenData(processedSample, listOfMarkerGenes, displayImage=True, pixelCombination='additive'):
    if 'tissueImageProcessed' in processedSample:
        geneImage = np.zeros_like(processedSample['tissueImageProcessed'])
    elif 'tissueImageRegistered' in processedSample:
        geneImage = np.zeros_like(processedSample['tissueImageRegistered'])
    if 'geneMatrixLog2' in processedSample:
        geneMatrix = processedSample['geneMatrixLog2']
    elif 'geneMatrixMasked' in processedSample:
        geneMatrix = processedSample['geneMatrixMasked']
    if 'geneList' in processedSample:
        listToSearch = processedSample['geneList']
    elif 'geneListMasked' in processedSample:
        listToSearch = processedSample['geneListMasked']
    if 'processedTissuePositionList' in processedSample:
        tissueCoordinates = processedSample['processedTissuePositionList']
    elif 'maskedTissuePositionList' in processedSample:
        tissueCoordinates = processedSample['maskedTissuePositionList']
    for actGene in enumerate(listOfMarkerGenes):
        geneIdx = listToSearch.index(actGene[1])
        geneArray = geneMatrix[geneIdx,:].todense()
        geneMask = np.squeeze(np.array(geneArray > 0))
        geneArray = geneArray[:,geneMask]
        geneArray = geneArray/np.max(geneArray)
        geneTissuePositions = tissueCoordinates[geneMask, :]
        genePixelNum = actGene[0] + 1
        for coordinate in enumerate(geneTissuePositions):
            xCoor = round(coordinate[1][1])
            yCoor = round(coordinate[1][0])
            if xCoor >= geneImage.shape[0]:
                xCoor = geneImage.shape[0] - 1
            if yCoor >= geneImage.shape[1]:
                yCoor = geneImage.shape[1] - 1
            if pixelCombination == 'additive':
                geneImage[xCoor, yCoor] += genePixelNum
            elif pixelCombination == 'replace':
                geneImage[xCoor, yCoor] = genePixelNum
            elif pixelCombination == 'binaryAdd':
                geneImage[xCoor, yCoor] += 1
            elif pixelCombination == 'geneExpressionScaledReplace':
                geneImage[xCoor, yCoor] = genePixelNum * geneArray[:,coordinate[0]]
            elif pixelCombination == 'geneExpressionScaledAdditive':
                geneImage[xCoor, yCoor] += genePixelNum * geneArray[:,coordinate[0]]
    geneImage = np.array(geneImage / np.max(geneImage) * 255, dtype='uint8')
    if displayImage==True:
        plt.figure()
        plt.imshow(geneImage, cmap='gray_r')
        plt.show()
    return geneImage


#%% load allen template from BrainGlobe

# allen CCF for merscope data is aligned to 25um
atlas='allen_mouse_25um'
ara_data = BrainGlobeAtlas(atlas, check_latest=False)
templateSlice = ara_data.reference[70,:,:]
plt.imshow(templateSlice)

#%%
"""
in order to make a 3d image, will need to round the ccf coordinates 
"""

roundedCoordinates = np.array([np.round(x_CCF), np.round(y_CCF), np.round(z_CCF)])

# find unique coordinates
uniqueCoordinates = np.unique(roundedCoordinates, axis=1)
del(x_CCF, y_CCF, z_CCF, tissue_coordinates, sample)
#%% find locations of duplicates
plt.close('all')
roundedCoordinates = roundedCoordinates.T

condensedGeneMatrixSavePath = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', 'condensedGeneMatrixFor3dVolume.npz')
    # check whether similarity matrix has already been calculated
if os.path.exists(condensedGeneMatrixSavePath):
    condensedGeneMatrix = sp_sparse.load_npz(condensedGeneMatrixSavePath)
else:
    # create empty gene matrix to assign new valuess to
    condensedGeneMatrix = np.zeros([uniqueCoordinates.shape[1], gene_matrix.shape[1]])
    for coor1 in range(uniqueCoordinates.shape[1]):
        uniqueIdx = np.where(np.all(roundedCoordinates == uniqueCoordinates[:,coor1], axis=1))[0]
        # leaving it as a mean for both multiple and singleton indices, since it won't effect singleton
        condensedGeneMatrix[coor1,:] = np.mean(gene_matrix[uniqueIdx,:], axis=0)
        # print(uniqueIdx)
    del(gene_matrix, roundedCoordinates)
    condensedGeneMatrix = sp_sparse.csr_array(condensedGeneMatrix)
    sp_sparse.save_npz(condensedGeneMatrixSavePath, condensedGeneMatrix)
