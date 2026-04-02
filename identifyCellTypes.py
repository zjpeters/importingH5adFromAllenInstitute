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
# from brainglobe_atlasapi import BrainGlobeAtlas
import sys
# sys.path.insert(0, os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code'))
# sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
# import stanly
import scipy.sparse as sp_sparse
# import scipy.spatial as sp_spatial
# import random
# import scipy
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score
import nibabel as nib
sys.path.insert(0, os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','code'))
# sys.path.insert(0,'/media/zjpeters/Expansion/merscopeDataFromAllenInstitute/code')
import allenMerscopeCode
import cv2
import nibabel as nib
import time
import scipy.stats as scipy_stats
import matplotlib.cm as cm
import matplotlib.patches as mpatches

# windows workstation locations
stanlyLoc = os.path.join('C:',os.sep, 'Users','onyh19ug', 'Documents', 'STANLY','code')
sourcedata = os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','sourcedata')
derivatives = os.path.join('D:',os.sep, 'merscopeDataFromAllenInstitute','derivatives')
# linux locations
# stanlyLoc = os.path.join('/', 'home', 'zjpeters', 'Documents', 'stanly', 'code')
# sourcedata = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata')
# derivatives = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives')

affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.25, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])

h5adLocation = os.path.join(sourcedata,'mouse_609882_registered.h5ad')
# list generated from selectGenePatterns, selecing only genes with strong patterns
listForGeneImage = ['Prdm12', 'Mal', 'Nts', 'Cbln1', 'Col1a1', 'Cdh13', 'Ramp1', 'Rgs6', 'Gpr88', 'Rorb', 'Slc17a7', 'Pou3f1', 'Zfpm2', 'Pvalb', 'Slc1a3']

listOfFiles = ['mouse_609882_registered.h5ad', 'mouse_609889_registered.h5ad', 
               'mouse_638850_registered.h5ad', 'mouse_658801_registered.h5ad', 
               'mouse_687997_registered.h5ad', 'mouse_702265_registered.h5ad']

housekeepingGenes = ['Gapdh', 'Actb', 'B2m', 'Hprt', 'Cyc1', 'Eif2a']

# create dictionary with information about different interneuron types
interneuron_information = dict.fromkeys(["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"])
for i in interneuron_information.keys():
    interneuron_information[i] = dict.fromkeys(['geneList', 'geneIdx'])
interneuron_information["Pvalb"]["geneList"] = ['Btbd11', 'Cntnap4', 'Eya4', 'Kcnmb2', 'Pvalb', 'Slit2']
interneuron_information["Sst"]["geneList"] = ['Calb1', 'Lypd6', 'Pdyn', 'Rab3b', 'Rbp4']
interneuron_information["Vip"]["geneList"] = ['Chat', 'Crh', 'Igf1', 'Penk', 'Pthlh', 'Sorcs3', 'Thsd7a', 'Vip']
interneuron_information["Sncg"]["geneList"] = ['Col19a1', 'Kctd12', 'Necab1', 'Slc44a5']
interneuron_information["Lamp5"]["geneList"] = ['Dner', 'Gad1', 'Gad2', 'Hapln1', 'Lamp5', 'Pde11a', 'Rasgrf2']

# load gene lists from paper
"""
identify genes present in allen data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join(stanlyLoc,'data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)

#%% calculate the order to store the slices
plt.close('all')
# first two images are displaying all NaN for ccf coordinates, so skipping for now
meanAllSamplesCCFZ = []
allImagesAllSamplesCCF = []
for filename in listOfFiles[2:]:
    f = h5py.File(os.path.join(sourcedata, filename))
    slice_codes = np.unique(f['obs']['section']['codes'][:])
    f.close()
    nSlices = len(np.unique(slice_codes))
    for i in range(nSlices + 1):
        sample = allenMerscopeCode.loadSingleSliceFromH5ad(os.path.join(sourcedata, filename), i)
        if ~np.any(np.isnan(sample['ccfCoordinates'])) and len(sample['ccfCoordinates'] > 0):
            meanAllSamplesCCFZ.append(np.mean(sample['ccfCoordinates'][:,0]))
    print(f'Completed checking {filename}')
imageSortIdx = np.argsort(meanAllSamplesCCFZ)
            
#%% look for housekeeping genes within sample

sample = allenMerscopeCode.loadSingleSliceFromDatedH5ad(h5adLocation, 10)

for gene in housekeepingGenes:
    try:
        print(sample['geneList'].index(gene))
    except ValueError:
        print(f'{gene} not in list')
        
"""
None of the original list appear in the allen data, will look over other resources
"""
#%% identify overlapping genes
interneuronGenesInMerscope = dict.fromkeys(["Pvalb", "Sst", "Vip", "Sncg", "Lamp5"])
for i in interneuronGenesInMerscope.keys():
    interneuronGenesInMerscope[i] = dict.fromkeys(['geneList', 'geneIdx'])
for interneuron_type in interneuronGenesInMerscope.keys():
    interneuron_gene_idx = []
    interneuron_gene_list = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        try:
            geneIdx = sample['geneList'].index(gene)
            interneuron_gene_list.append(gene)
            interneuron_gene_idx.append(geneIdx)
        except ValueError:
            print('Gene not in list')
    interneuronGenesInMerscope[interneuron_type]['geneIdx'] = np.array(interneuron_gene_idx)
    interneuronGenesInMerscope[interneuron_type]['geneList'] = interneuron_gene_list
    
#%%
# casefoldGeneList = []
# for gene in sample['geneList']:
#     casefoldGeneList.append(gene.casefold())
# casefoldGeneList = list(casefoldGeneList)
cellTypeCasefoldList = []
for gene in cellTypeGeneExpressionList['gene']:
    cellTypeCasefoldList.append(gene.casefold())

sampleCasefoldList = []
for gene in sample['geneList']:
    sampleCasefoldList.append(gene.casefold())

cellTypeGeneIdx = [x in sampleCasefoldList for x in cellTypeCasefoldList]

cellTypeGenesInSample = cellTypeGeneExpressionList[cellTypeGeneIdx]
# create lists of cell type genes
cellTypes = np.unique(cellTypeGenesInSample['Celltype'])
cellTypeGeneLists = {}
for i in cellTypes:
    cellTypeDF = cellTypeGenesInSample[cellTypeGenesInSample['Celltype'] == i]
    singleCellTypeGeneList = []
    for j in cellTypeDF['gene']:
        try:
            geneIdx = sampleCasefoldList.index(j.casefold())
            singleCellTypeGeneList.append([sample['geneList'][geneIdx], geneIdx])
            cellTypeGeneLists[i] = np.array(singleCellTypeGeneList)

        except ValueError:
            # code above should work well, though might need to consider if there
            # are situations where a casefold gene name would lead to duplicates
            print('Gene not found')

for interneuron_type in interneuron_information.keys():
    singleCellTypeGeneList = []
    for gene in interneuron_information[interneuron_type]['geneList']:
        try:
            geneIdx = sample['geneList'].index(gene)
            singleCellTypeGeneList.append([gene, geneIdx])
            cellTypeGeneLists[interneuron_type] = np.array(singleCellTypeGeneList)
        except ValueError:
            print('Gene not in list')
    
# for i in interneuron_information:
#     singleCellTypeGeneList = np.array([interneuron_information[i]['geneList'], interneuron_information[i]['geneIdx']])
#     cellTypeGeneLists[i] = singleCellTypeGeneList.T
#%% try to identify cell types within data
geneMatrixZScore = sample['geneMatrix'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
plt.close('all')
meanZScoreMatrixCellTypes = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
cellTypeProbs = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
for i in range(sample['geneMatrix'].shape[0]):
    for j in enumerate(cellTypeGeneLists): 
        geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[i, geneMask]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixCellTypes[i, j[0]] = meanZScore
        # for the sake of calculating cell type probability, we don't want negative z-scores
        if meanZScore < 0:
            cellTypeProbs[i,j[0]] = 0
        else:
            cellTypeProbs[i,j[0]] = scipy_stats.norm.cdf(np.abs(meanZScore))
percentIDed = np.sum(np.any(cellTypeProbs > 0, axis=1))/len(cellTypeProbs)

# meanZScoreMatrixInterneurons = np.zeros([sample['geneMatrix'].shape[1], len(cellTypes)])
# for i in range(sample['geneMatrix'].shape[1]):
#     for j in enumerate(interneuron_information): 
#         geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
#         cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i]))
#         # cellTypeMatrix = cellTypeMatrix[geneMask, :]
#         # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
#         meanZScore = np.mean(cellTypeMatrix)
#         meanZScoreMatrixInterneurons[i, j[0]] = meanZScore

#%% try the same as above but selecting the maximum z-score from the cell type gene matrix

plt.close('all')
maxZScoreMatrixCellTypes = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
cellTypeProbsMax = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
for i in range(sample['geneMatrix'].shape[0]):
    for j in enumerate(cellTypeGeneLists): 
        geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[i, geneMask]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        maxZScore = np.max(cellTypeMatrix)
        maxZScoreMatrixCellTypes[i, j[0]] = maxZScore
        # for the sake of calculating cell type probability, we don't want negative z-scores
        if maxZScore < 0:
            cellTypeProbsMax[i,j[0]] = 0
        else:
            cellTypeProbsMax[i,j[0]] = scipy_stats.norm.cdf(np.abs(maxZScore))

#%% select top scoring cell type
posIDMask = np.any(cellTypeProbs > 0, axis=1)
cellTypeProbsPosID = cellTypeProbs[posIDMask]
maxCellTypeIdx = np.argmax(cellTypeProbsPosID, axis=1)

#%% display cell type ints
# create color profile
# astColor = cm.tab10(0)
# endColor = cm.tab10(1)
# micColor = cm.tab10(2)
# neuColor = cm.tab20(7)
# oliColor = cm.tab10(4)
astColor = cm.tab10(0)
endColor = cm.tab10(1)
micColor = cm.tab10(2)
neuColor = cm.tab10(3)
oliColor = cm.tab10(4)

sampleColors = np.zeros([maxCellTypeIdx.shape[0], 4])
astIdx = np.where(maxCellTypeIdx == 0)
sampleColors[astIdx,:] = astColor
endIdx = np.where(maxCellTypeIdx == 1)
sampleColors[endIdx,:] = endColor
micIdx = np.where(maxCellTypeIdx == 2)
sampleColors[micIdx,:] = micColor
neuIdx = np.where(maxCellTypeIdx == 3)
sampleColors[neuIdx,:] = neuColor
oliIdx = np.where(maxCellTypeIdx == 4)
sampleColors[oliIdx,:] = oliColor
# display data
plt.close('all')
fig = plt.figure(figsize=(10,8))
ax= fig.add_subplot()
geneScatter = ax.scatter(sample['ccfCoordinates'][posIDMask,2], sample['ccfCoordinates'][posIDMask,1], c=sampleColors, s=1, linewidth=0)
ax.yaxis.set_inverted(True)
handles, labels = ax.get_legend_handles_labels()
patch = mpatches.Patch(color=astColor, label='astrocytes')
handles.append(patch) 
patch = mpatches.Patch(color=endColor, label='endothelial cells')
handles.append(patch) 
patch = mpatches.Patch(color=micColor, label='microglia')
handles.append(patch) 
patch = mpatches.Patch(color=neuColor, label='neurons')
handles.append(patch) 
patch = mpatches.Patch(color=oliColor, label='oligodendrocytes')
handles.append(patch) 
ax.legend(handles=handles,bbox_to_anchor=(0.8, 1))
ax.set_aspect('equal')
ax.axis('off')
plt.show()
# plt.savefig(os.path.join(derivatives, f'testImage.svg'), bbox_inches='tight', dpi=300)
#%% loop over samples and identify cell types in each then save image
for filename in enumerate(listOfFiles[2:]):
    print(f'Processing {filename[1]}')
    f = h5py.File(os.path.join(sourcedata, filename[1]))
    slice_codes = np.unique(f['obs']['section']['codes'][:])
    f.close()
    filenameForSaving = filename[1].split(".")[0]
    for sliceCode in range(len(slice_codes)):
        if os.path.exists(os.path.join(derivatives, f'cellTypeLabelling_sample{filenameForSaving}_slice{sliceCode}.png')):
            print(f'cellTypeLabelling_sample{filenameForSaving}_slice{sliceCode}.svg')
        else:
            sample = allenMerscopeCode.loadSingleSliceFromH5ad(os.path.join(sourcedata, filename[1]), sliceCode)
            geneMatrixZScore = sample['geneMatrix'].todense()
            geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
            maxZScoreMatrixCellTypes = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
            cellTypeProbsMax = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
            for i in range(sample['geneMatrix'].shape[0]):
                for j in enumerate(cellTypeGeneLists): 
                    geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
                    cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[i, geneMask]))
                    # cellTypeMatrix = cellTypeMatrix[geneMask, :]
                    # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
                    maxZScore = np.max(cellTypeMatrix)
                    maxZScoreMatrixCellTypes[i, j[0]] = maxZScore
                    # for the sake of calculating cell type probability, we don't want negative z-scores
                    if maxZScore < 0:
                        cellTypeProbsMax[i,j[0]] = 0
                    else:
                        cellTypeProbsMax[i,j[0]] = scipy_stats.norm.cdf(np.abs(maxZScore))
            posIDMask = np.any(cellTypeProbsMax > 0, axis=1)
            cellTypeProbsPosID = cellTypeProbsMax[posIDMask]
            maxCellTypeIdx = np.argmax(cellTypeProbsPosID, axis=1)
            
            sampleColors = np.zeros([maxCellTypeIdx.shape[0], 4])
            astIdx = np.where(maxCellTypeIdx == 0)
            sampleColors[astIdx,:] = astColor
            endIdx = np.where(maxCellTypeIdx == 1)
            sampleColors[endIdx,:] = endColor
            micIdx = np.where(maxCellTypeIdx == 2)
            sampleColors[micIdx,:] = micColor
            neuIdx = np.where(maxCellTypeIdx == 3)
            sampleColors[neuIdx,:] = neuColor
            oliIdx = np.where(maxCellTypeIdx == 4)
            sampleColors[oliIdx,:] = oliColor
            # display data
            plt.close('all')
            fig = plt.figure(figsize=(10,8))
            ax= fig.add_subplot()
            geneScatter = ax.scatter(sample['ccfCoordinates'][posIDMask,2], sample['ccfCoordinates'][posIDMask,1], c=sampleColors, s=1, linewidth=0)
            ax.yaxis.set_inverted(True)
            handles, labels = ax.get_legend_handles_labels()
            patch = mpatches.Patch(color=astColor, label='astrocytes')
            handles.append(patch) 
            patch = mpatches.Patch(color=endColor, label='endothelial cells')
            handles.append(patch) 
            patch = mpatches.Patch(color=micColor, label='microglia')
            handles.append(patch) 
            patch = mpatches.Patch(color=neuColor, label='neurons')
            handles.append(patch) 
            patch = mpatches.Patch(color=oliColor, label='oligodendrocytes')
            handles.append(patch) 
            ax.legend(handles=handles,bbox_to_anchor=(0.8, 1))
            ax.set_aspect('equal')
            ax.axis('off')
            plt.show()
            plt.savefig(os.path.join(derivatives, f'cellTypeLabelling_sample{filenameForSaving}_slice{sliceCode}.png'), bbox_inches='tight', dpi=300)
            
#%% create images where the cell types are isolated, also try to create volumes for each cell type
imageVolume = np.zeros([360,480,330])
imageVolumeDict = {}
for i in range(len(cellTypes)):
    imageVolumeDict[i] = np.zeros([360,480,330])
sliceNumber = 0
"""
only run if needing to recreate data, otherwise it takes awhile
"""
for filename in enumerate(listOfFiles[2:]):
    print(f'Processing {filename[1]}')
    f = h5py.File(os.path.join(sourcedata, filename[1]))
    slice_codes = np.unique(f['obs']['section']['codes'][:])
    f.close()
    filenameForSaving = filename[1].split(".")[0]
    for sliceCode in range(len(slice_codes)):
        # if os.path.exists(os.path.join(derivatives, f'cellTypeLabelling_sample{filenameForSaving}_slice{sliceCode}.svg')):
        #     print(f'cellTypeLabelling_sample{filenameForSaving}_slice{sliceCode}.svg')
        # else:
        sample = allenMerscopeCode.loadSingleSliceFromH5ad(os.path.join(sourcedata, filename[1]), sliceCode)
        if ~np.any(np.isnan(sample['ccfCoordinates'])) and len(sample['ccfCoordinates'] > 0):
            geneMatrixZScore = sample['geneMatrix'].todense()
            geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
            maxZScoreMatrixCellTypes = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
            cellTypeProbsMax = np.zeros([sample['geneMatrix'].shape[0], len(cellTypeGeneLists)])
            for cell in range(sample['geneMatrix'].shape[0]):
                for cellType in enumerate(cellTypeGeneLists): 
                    geneMask = np.array(cellTypeGeneLists[cellType[1]][:,1], dtype='int32')
                    cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[cell, geneMask]))
                    # cellTypeMatrix = cellTypeMatrix[geneMask, :]
                    # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
                    maxZScore = np.max(cellTypeMatrix)
                    maxZScoreMatrixCellTypes[cell, cellType[0]] = maxZScore
                    # for the sake of calculating cell type probability, we don't want negative z-scores
                    if maxZScore < 0:
                        cellTypeProbsMax[cell,cellType[0]] = 0
                    else:
                        cellTypeProbsMax[cell,cellType[0]] = scipy_stats.norm.cdf(np.abs(maxZScore))
            posIDMask = np.any(cellTypeProbsMax > 0, axis=1)
            posIDCoors = np.array([sample['ccfCoordinates'][posIDMask,2], sample['ccfCoordinates'][posIDMask,1]]).T
            cellTypeProbsPosID = cellTypeProbsMax[posIDMask]
            maxCellTypeIdx = np.argmax(cellTypeProbsPosID, axis=1)
            
            for cellType in np.unique(maxCellTypeIdx):
                cellTypeIdx = np.where(maxCellTypeIdx == cellType)[0]
                cellTypeCoors = np.array(np.floor(posIDCoors[cellTypeIdx,:]), dtype='int32')
                for coordinate in cellTypeCoors:
                    xCoor = coordinate[0]
                    yCoor = coordinate[1]
                    imageVolumeDict[cellType][yCoor, xCoor, sliceNumber] = 1
                # fig = plt.figure(figsize=(10,8))
                # ax= fig.add_subplot()    
                # geneScatter = ax.scatter(posIDCoors[cellTypeIdx,0], posIDCoors[cellTypeIdx,1], s=2)
                # ax.yaxis.set_inverted(True)
                # plt.show()
                #### version below overlays all cell types
                # fig = plt.figure(figsize=(10,8))
                # ax= fig.add_subplot()
                # for cellType in np.unique(maxCellTypeIdx):
                #     cellTypeIdx = np.where(maxCellTypeIdx == cellType)[0]
                    
                #     geneScatter = ax.scatter(posIDCoors[cellTypeIdx,0], posIDCoors[cellTypeIdx,1], s=2)
                #     ax.yaxis.set_inverted(True)
                # plt.show()
            sliceNumber +=1
            print(f"Slice number {sliceCode} completed")
# save unsorted volumes as 1D arrays that can be reconstructed
for i in range(len(imageVolumeDict)):
    np.savetxt(os.path.join(derivatives, f'unsorted_cell_type_image_volume_{i}.csv'), imageVolumeDict[i].reshape(imageVolumeDict[i].shape[0], -1), delimiter=',')
    
#%% load unsorted volumes
imageVolumeDict = {}
for i in range(5):
    print(f'Opening unsorted_cell_type_image_volume_{i}.csv')
    imageVolumeDict[i] = np.loadtxt(os.path.join(derivatives, f'unsorted_cell_type_image_volume_{i}.csv'), delimiter=',')
    print(f'finished loading unsorted_cell_type_image_volume_{i}.csv')
#%% restructure data into original shape    
for i in range(len(imageVolumeDict)):
    imageVolumeDict[i] = imageVolumeDict[i].reshape(360, 480, 330)
    
#%% sort images 
sortedImageVolumeDict = {}
for i in range(len(cellTypes)):
    sortedImageVolumeDict[i] = np.zeros([360,480,216])
    
for i in enumerate(imageSortIdx):
    print(i)
    for cellType in range(len(imageVolumeDict)):
        sortedImageVolumeDict[cellType][:,:,i[0]] = imageVolumeDict[cellType][:,:,i[1]]
        
#%% create nifti volumes
affMatrix = np.array([[-0, -0, 0.075, -5.7],[-0.025, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])
affMatrix = np.array([[-0, -0, 0.025, -5.7],[-0.025, -0, -0, 5.3], [0, -0.025, 0, 5.175], [0,0,0,1]])

for i in range(len(sortedImageVolumeDict)):    
    niiImage = nib.Nifti1Image(sortedImageVolumeDict[i], affMatrix)
    nib.save(niiImage, os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','derivatives', f'cellTypeVolume_{cellTypes[i]}.nii'))

#%% find sorting order of cell types
for i in range(len(sortedImageVolumeDict)):
    cellSum = np.sum(sortedImageVolumeDict[i])
    print(f'Number of {cellTypes[i]}: {cellSum}')