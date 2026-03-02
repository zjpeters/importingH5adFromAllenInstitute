#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 07:19:47 2026

@author: zjpeters
"""

import os
import pandas as pd
import h5py
# from scipy.sparse import csr_matrix
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
import scipy.sparse as sp_sparse
from scipy.sparse import csr_matrix
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

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

# create function to display gene image from single slice
# def displaySingleSliceGeneImage(sliceNumber, geneMatrix, geneList, micronsToDisplay=25, pixelCombination='additive', displayImage=True):
#     sliceIdx = np.where(slice_codes == sliceNumber)[0]
#     slice_coordinates = tissue_coordinates[sliceIdx, :]/micronsToDisplay
#     image_size = np.round(np.max(tissue_coordinates, axis=0))/micronsToDisplay
#     gene_image = np.zeros([int(image_size[0])+1, int(image_size[1])+1])
#     for actGene in enumerate(geneList):
#         geneArray = geneMatrix[sliceIdx,:].todense()[:, [actGene[0]]]
#         geneArray = geneArray/np.max(geneArray)
#         geneMask = np.squeeze(np.array(geneArray > 0))
#         geneCoordinates = slice_coordinates[geneMask,:]
#         genePixelNum = actGene[0] + 1
#         for coordinate in enumerate(geneCoordinates):
#             xCoor = int(coordinate[1][0])
#             yCoor = int(coordinate[1][1])
#             # print(coordinate[1], xCoor)
#             if pixelCombination == 'additive':
#                 # print(xCoor, yCoor, genePixelNum)
#                 gene_image[xCoor, yCoor] += genePixelNum
#             elif pixelCombination == 'replace':
#                 gene_image[xCoor, yCoor] = genePixelNum
#             elif pixelCombination == 'geneExpressionScaledReplace':
#                 gene_image[xCoor, yCoor] = genePixelNum * geneArray[:,coordinate[0]]
#             elif pixelCombination == 'geneExpressionScaledAdditive':
#                 gene_image[xCoor, yCoor] += genePixelNum * geneArray[coordinate[0]]
#             elif pixelCombination == 'geneExpression':
#                 gene_image[xCoor, yCoor] += geneArray[coordinate[0]]
#     gene_image = np.array(gene_image / np.max(gene_image) * 255, dtype='uint8')
#     if displayImage == True:
#         plt.figure()
#         plt.imshow(gene_image, cmap='gray')    
#         # plt.scatter(slice_coordinates[:,0], slice_coordinates[:,1], s=1)
#         plt.show()
#     return gene_image
sample = loadSingleSliceFromH5ad(fileLocation, 4, displayScatter=True)
#%% test updated selectGenePatterns using allen data
sample['derivativesPath'] = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives')
sample['sampleID'] = 'slice_4'
topGeneList = stanly.selectGenePatterns(sample, k=25, nRandomCells=1000)

#%% reduce gene matrix to only genes of interest
gene_list_idx = []
for gene_name in topGeneList:
    gene_list_idx.append(topGeneList.index(gene_name))
gene_list_idx = np.array(gene_list_idx)

short_gene_matrix = sample['geneMatrix'][:,gene_list_idx]
#%% display genes
plt.close('all')
for idx, gene in enumerate(topGeneList):
    genePosMask = short_gene_matrix[:,idx] > 0
    # displayCoors = sample['ccfCoordinates'][genePosMask,2], sample['ccfCoordinates'][genePosMask,1]
    displayColors = np.squeeze(np.array(short_gene_matrix[:,idx].todense()))
    fig,ax = plt.subplots(1,1, figsize=(12,8))
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=displayColors, cmap='Reds', s=2)
    ax.yaxis.set_inverted(True)
    plt.title(gene)
    plt.show()

#%% use list of genes from one map paper
"""
Found list of genes from "Cerebellar cortical organization: a one-map hypothesis"
that were present in the allen gene list, displaying them below
"""
geneMatrixZScore = sample['geneMatrix'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
cerebellarGeneList = ['Zeb2', 'Ebf2','Grm1','Reln','Epha5','Cdh7','Cdh9','Cdh12','Cdh13','Cdh20', 'Pcp4l1']
plt.close('all')
backgroundCells = np.zeros([sample['ccfCoordinates'].shape[0],3])
backgroundCells[:] = [0.5, 0.5, 0.5]
for idx, gene in enumerate(cerebellarGeneList):
    geneIdx = sample['geneList'].index(gene)
    geneArray = np.squeeze(np.array(geneMatrixZScore[:,geneIdx]))
    geneArraySort = np.sort(geneArray)
    geneSortIdx = np.argsort(geneArray)
    geneCoorsSort = sample['ccfCoordinates'][geneSortIdx, :]
    genePosMask = geneArraySort > 0
    geneCoors = geneCoorsSort[genePosMask,:]
    # displayCoors = sample['ccfCoordinates'][genePosMask,2], sample['ccfCoordinates'][genePosMask,1]
    # displayColors = np.squeeze(np.array(short_gene_matrix[:,idx].todense()))
    fig,ax = plt.subplots(1,1, figsize=(12,8))
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=backgroundCells)
    ax.scatter(geneCoors[:,2], geneCoors[:,1], c=geneArraySort[genePosMask], cmap='seismic', vmin=-4, vmax=4, s=2)
    ax.yaxis.set_inverted(True)
    plt.title(gene)
    plt.show()

#%% perform same plotting, but including low z scores

geneMatrixZScore = sample['geneMatrix'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
cerebellarGeneList = ['Zeb2', 'Ebf2','Grm1','Reln','Epha5','Cdh7','Cdh9','Cdh12','Cdh13','Cdh20']
plt.close('all')
backgroundCells = np.zeros([sample['ccfCoordinates'].shape[0],3])
backgroundCells[:] = [0.5, 0.5, 0.5]
for idx, gene in enumerate(cerebellarGeneList):
    geneIdx = sample['geneList'].index(gene)
    geneArray = np.squeeze(np.array(geneMatrixZScore[:,geneIdx]))
    geneArraySort = np.sort(geneArray)
    geneSortIdx = np.argsort(geneArray)
    geneCoorsSort = sample['ccfCoordinates'][geneSortIdx, :]
    # genePosMask = geneArraySort > 0
    # geneCoors = geneCoorsSort[genePosMask,:]
    # displayCoors = sample['ccfCoordinates'][genePosMask,2], sample['ccfCoordinates'][genePosMask,1]
    # displayColors = np.squeeze(np.array(short_gene_matrix[:,idx].todense()))
    fig,ax = plt.subplots(1,1, figsize=(12,8))
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=backgroundCells)
    ax.scatter(geneCoorsSort[:,2], geneCoorsSort[:,1], c=geneArraySort, cmap='seismic', vmin=-2, vmax=2, s=2)
    ax.yaxis.set_inverted(True)
    plt.title(gene)
    plt.show()   
    
#%% run correlations for genes and Zeb2
zeb2Idx = sample['geneList'].index('Zeb2')
zeb2Array = np.squeeze(np.array(sample['geneMatrix'][:,zeb2Idx].todense()))
zeb2Mask = zeb2Array > 0
secondGeneArray = np.squeeze(np.array(sample['geneMatrix'][:,0].todense()))
x = np.corrcoef(zeb2Array, secondGeneArray)
corrList = []
for i in range(len(sample['geneList'])):
    secondGeneArray = np.squeeze(np.array(sample['geneMatrix'][:,i].todense()))
    x = np.corrcoef(zeb2Array, secondGeneArray)
    corrList.append(x[0,1])

sortedCorrs = np.sort(np.array(corrList))
sortedCorrsIdx = np.argsort(np.array(corrList))

#%% plot genes with high Zeb2 correlation

plt.close('all')
for idx in range(701, 686, -1):
    geneIdx = sortedCorrsIdx[idx]
    genePosMask = sample['geneMatrix'][:,geneIdx] > 0
    # displayCoors = sample['ccfCoordinates'][genePosMask,2], sample['ccfCoordinates'][genePosMask,1]
    displayColors = np.squeeze(np.array(sample['geneMatrix'][:,geneIdx].todense()))
    fig,ax = plt.subplots(1,1, figsize=(12,8))
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=displayColors, cmap='Reds', s=2)
    ax.yaxis.set_inverted(True)
    plt.title(sample['geneList'][geneIdx])
    plt.show()

#%% print top correlating genes
for idx in range(701, 686, -1):
    geneIdx = sortedCorrsIdx[idx]
    print(sample['geneList'][geneIdx])
#%% create figure showing all of the above gene expressions combined 

plt.close('all')
combinedGeneExpression = np.zeros([sample['geneMatrix'].shape[0],1])
for idx, gene in enumerate(cerebellarGeneList):
    geneIdx = sample['geneList'].index(gene)
    geneArray = geneMatrixZScore[:,geneIdx]
    combinedGeneExpression += geneArray

combinedGeneExpression = np.squeeze(np.array(combinedGeneExpression))
combinedGeneSort = np.sort(combinedGeneExpression)
combinedGeneSortIdx = np.argsort(combinedGeneExpression)
geneCoorsSort = sample['ccfCoordinates'][combinedGeneSortIdx, :]
genePosMask = combinedGeneSort > 0
geneCoors = geneCoorsSort[genePosMask,:]
fig,ax = plt.subplots(1,1, figsize=(12,8))
ax.scatter(geneCoorsSort[:,2], geneCoorsSort[:,1], c=combinedGeneSort, cmap='seismic', s=2)
ax.yaxis.set_inverted(True)
plt.title("Combination of z-scores for:\n 'Zeb2', 'Ebf2','Grm1','Reln','Epha5','Cdh7','Cdh9','Cdh12','Cdh13','Cdh20'")
plt.show()

    


#%% calculate the cosine similarity for single cerebellar slice
simMatrixSavePath = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives','slice_4_similarity_matrix.npz')
similarityMatrix = stanly.measureTranscriptomicSimilarity(sample['geneMatrix'], savePath=simMatrixSavePath, axis=0, nRandomCells=6000, edgeList='RandomCells')
sp_sparse.save_npz(simMatrixSavePath, similarityMatrix)
#%% cluster using similarity table
# can start by using number of genes
n_eigenvectors = 500
nDigitalSpots = sample['geneMatrix'].shape[0]
k = 15

Wcontrol = similarityMatrix.todense()

#% create laplacian for control
Wcontrol = (Wcontrol - np.nanmin(Wcontrol))/(np.nanmax(Wcontrol) - np.nanmin(Wcontrol))
Wcontrol[Wcontrol==1] = 0
# Wcontrol[np.isnan(Wcontrol)] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = scipy.sparse.linalg.eigs(Lcontrol, k=n_eigenvectors)
eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])

# run sillhoutte analysis on control clustering
    
# # Create a subplot with 1 row and 2 columns
# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_size_inches(18, 7)

# # The 1st subplot is the silhouette plot
# # The silhouette coefficient can range from -1, 1 but in this example all
# # lie within [-0.1, 1]
# ax1.set_xlim([-1, 1])
# # The (n_clusters+1)*10 is for inserting blank space between silhouette
# # plots of individual clusters, to demarcate them clearly.
# ax1.set_ylim([0, nDigitalSpots + (k + 1) * 10])

clusters = KMeans(n_clusters=k, init='random', n_init=500, tol=1e-8,)
cluster_labels = clusters.fit_predict(np.real(np.array(eigvecControlSort)[:,0:k+1]))

#%%
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
silhouette_avg = silhouette_score(np.real(np.array(eigvecControlSort)[:,0:k]), cluster_labels)
print(
    "For n_clusters =",
    k,
    "The average silhouette_score is :",
    silhouette_avg,
)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(np.real(np.array(eigvecControlSort)[:,0:k]), cluster_labels)

y_lower = 10
for i in range(k):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.tab20b(float(i) / k)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.tab20b(cluster_labels.astype(float) / k)
ax2.imshow(sampleToCluster['tissueImageProcessed'],cmap='gray_r')
ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors, s=5)
ax2.set_title("The visualization of the clustered control data.")
ax2.axis('off')

plt.suptitle(
    "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
    % k,
    fontsize=14,
    fontweight="bold",
)