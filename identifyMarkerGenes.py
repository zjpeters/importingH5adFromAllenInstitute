#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:42:31 2025

@author: zjpeters
"""
import os
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as scipy_stats
"""
identify genes present in allen data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join('/','home','zjpeters','Documents','stanly','code','data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)
fileLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')

#%% load registered h5ad file and print keys
h5adLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')
def loadGeneListFromH5ad(h5adLocation):
    # builds a csr matrix from data in the h5ad file
    f = h5py.File(h5adLocation)
    # x_attrs = dict(f['X'].attrs)
    # gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
    gene_list = np.array(f['var']['_index'], dtype=str)
    f.close()
    return list(gene_list)

gene_list = loadGeneListFromH5ad(h5adLocation)
#%% combine loading of matrix and coordinates for single slice

def loadSingleSliceFromH5ad(h5adLocation, sliceNumber, displayScatter=False):
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
sample = loadSingleSliceFromH5ad(fileLocation, 18, displayScatter=True)

#%% 
plt.close('all')
for i in range(10, len(sample['geneList']), 50):
    fig,ax = plt.subplots(1,1)
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=np.array(sample['geneMatrix'][:,i].todense()), s=1, cmap='Reds')
    ax.yaxis.set_inverted(True)
    plt.title(sample['geneList'][i])
    plt.show()

#%%
### using the code below, double checked that the center line for the hemispheres is at 228
plt.close('all')

geneIdx = sample['geneList'].index('Ramp1')
for i in range(10,50, 5):
    sample = loadSingleSliceFromH5ad(fileLocation, i, displayScatter=True)
    fig,ax = plt.subplots(1,1)
    ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1], c=np.array(sample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
    ax.yaxis.set_inverted(True)
    plt.title(sample['geneList'][geneIdx])
    plt.show()

#%% write function that separates a sample into right and left hemispheres

# leftCoorsIdx = sample['ccfCoordinates'][:,2] < 228
# rightCoorsIdx = sample['ccfCoordinates'][:,2] > 228
# leftCoors = sample['ccfCoordinates'][leftCoorsIdx,:]
# rightCoors = sample['ccfCoordinates'][rightCoorsIdx,:]

def splitHemispheres(sampleToSplit):
    leftCoorsIdx = sample['ccfCoordinates'][:,2] < 228
    rightCoorsIdx = sample['ccfCoordinates'][:,2] > 228
    leftSample = {}
    rightSample = {}
    leftSample['ccfCoordinates'] = sample['ccfCoordinates'][leftCoorsIdx,:]
    rightSample['ccfCoordinates'] = sample['ccfCoordinates'][rightCoorsIdx,:]
    leftSample['geneMatrix'] = sample['geneMatrix'][leftCoorsIdx,:]
    rightSample['geneMatrix'] = sample['geneMatrix'][rightCoorsIdx,:]
    leftSample['tissuePositionCoordinates'] = sample['tissuePositionCoordinates'][leftCoorsIdx,:]
    rightSample['tissuePositionCoordinates'] = sample['tissuePositionCoordinates'][rightCoorsIdx,:]
    leftSample['geneList'] = sample['geneList']
    rightSample['geneList'] = sample['geneList']
    return leftSample, rightSample

leftSample, rightSample = splitHemispheres(sample)

fig,ax = plt.subplots(1,1)
ax.scatter(leftSample['ccfCoordinates'][:,2], leftSample['ccfCoordinates'][:,1], c=np.array(leftSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
ax.yaxis.set_inverted(True)
plt.title(leftSample['geneList'][geneIdx])
plt.show()

fig,ax = plt.subplots(1,1)
ax.scatter(rightSample['ccfCoordinates'][:,2], rightSample['ccfCoordinates'][:,1], c=np.array(rightSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
ax.yaxis.set_inverted(True)
plt.title(rightSample['geneList'][geneIdx])
plt.show()

#%%

# sample = loadSingleSliceFromH5ad(fileLocation, 20, displayScatter=True)
# leftSample, rightSample = splitHemispheres(sample)

# fig,ax = plt.subplots(1,1)
# ax.scatter(leftSample['ccfCoordinates'][:,2], leftSample['ccfCoordinates'][:,1], c=np.array(leftSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
# ax.yaxis.set_inverted(True)
# plt.title(leftSample['geneList'][geneIdx])
# plt.show()

# fig,ax = plt.subplots(1,1)
# ax.scatter(rightSample['ccfCoordinates'][:,2], rightSample['ccfCoordinates'][:,1], c=np.array(rightSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
# ax.yaxis.set_inverted(True)
# plt.title(rightSample['geneList'][geneIdx])
# plt.show()

#%% check for t-statistic between hemispheres
tStats = []
pVals = []
plt.close('all')
for i in range(len(sample['geneList'])):
    ttest = scipy_stats.ttest_ind(np.squeeze(np.sort(np.array(leftSample['geneMatrix'][:,i].todense()))), np.sort(np.squeeze(np.array(rightSample['geneMatrix'][:,i].todense()))))
    tStats.append(ttest[0])
    pVals.append(ttest[1])
tStats = np.array(tStats)
pVals = np.array(pVals)
#%% sort pVals and look at top 10 most significant
plt.close('all')
sortIdx = np.argsort(pVals)
for i in sortIdx[0:10]:
    fig,ax = plt.subplots(1,2)
    ax[0].scatter(leftSample['ccfCoordinates'][:,2], leftSample['ccfCoordinates'][:,1], c=np.array(leftSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
    ax[0].yaxis.set_inverted(True)
    ax[0].set_title(np.mean(np.array(leftSample['geneMatrix'][:,i].todense())))
    ax[1].scatter(rightSample['ccfCoordinates'][:,2], rightSample['ccfCoordinates'][:,1], c=np.array(rightSample['geneMatrix'][:,geneIdx].todense()), s=1, cmap='Reds')
    ax[1].yaxis.set_inverted(True)
    ax[1].set_title(np.mean(np.array(rightSample['geneMatrix'][:,i].todense())))
    plt.suptitle(f"{rightSample['geneList'][i]}, t-stat={tStats[i]}")
    plt.show()
#%% loop over gene list from allen and find matching genes in cell type specificity list

cellTypeCasefoldList = []
for gene in cellTypeGeneExpressionList['gene']:
    cellTypeCasefoldList.append(gene.casefold())

allenCasefoldList = []
for gene in gene_list:
    allenCasefoldList.append(gene.casefold())
    
#%%

cellTypeGeneIdx = [x in allenCasefoldList for x in cellTypeCasefoldList]

cellTypeGenesInAllen = cellTypeGeneExpressionList[cellTypeGeneIdx]

#%% create dictionary with information about different interneuron types, from csv from Junko
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
identify cell type genes present in data that are also present in:
    "Brain Cell Type Specific Gene Expression and Co-expression Network Architectures"
    https://www.nature.com/articles/s41598-018-27293-5
"""
cellTypeSpreadsheetLocation = os.path.join('/','home','zjpeters','Documents','stanly','code','data','cellTypeMarkerGeneInfo','Brain Cell Type Specific Gene Expression and Co-expression Network Architectures_41598_2018_27293_MOESM2_ESM_mouse_specificity.csv')
cellTypeGeneExpressionList = pd.read_csv(cellTypeSpreadsheetLocation)

#%% identify overlapping genes
for interneuron_type in interneuron_information.keys():
    interneuron_gene_idx = np.empty_like(interneuron_information[interneuron_type]['geneList'])
    i = 0
    for gene in interneuron_information[interneuron_type]['geneList']:
        try:
            geneIdx = sample['geneList'].index(gene)
            interneuron_gene_idx[i] = geneIdx
        except ValueError:
            # code above should work well, though might need to consider if there
            # are situations where a casefold gene name would lead to duplicates
            print('Gene not found')
        i += 1
    interneuron_information[interneuron_type]['geneIdx'] = np.array(interneuron_gene_idx)
    
#%% loop over gene list from data and find matching genes in cell type specificity list
casefoldGeneList = []
for gene in sample['geneList']:
    casefoldGeneList.append(gene.casefold())
xeniumCasefoldGeneList = list(casefoldGeneList)
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
            geneIdx = xeniumCasefoldGeneList.index(j.casefold())
            singleCellTypeGeneList.append([sample['geneList'][geneIdx], geneIdx])
            cellTypeGeneLists[i] = np.array(singleCellTypeGeneList)

        except ValueError:
            # code above should work well, though might need to consider if there
            # are situations where a casefold gene name would lead to duplicates
            print('Gene not found')

for i in interneuron_information:
    singleCellTypeGeneList = np.array([interneuron_information[i]['geneList'], interneuron_information[i]['geneIdx']])
    cellTypeGeneLists[i] = singleCellTypeGeneList.T

#%% use gene lists to identify cell type of each cell
"""
neurons have the largest number of genes included, would explain the low z-score
this is becasuse of the variety of neurons, and should be somewhat fixed by 
inclusion of interneurons, but will still need to consider other options
"""
geneMatrixZScore = sample['geneMatrix'].todense()
geneMatrixZScore = (geneMatrixZScore - np.mean(geneMatrixZScore, axis=0))/np.std(geneMatrixZScore, axis=0)
plt.close('all')
meanZScoreMatrixCellTypes = np.zeros([sample['geneMatrix'].shape[1], len(cellTypeGeneLists)])
cellTypeProbs = np.zeros([sample['geneMatrix'].shape[1], len(cellTypeGeneLists)])
for i in range(sample['geneMatrix'].shape[1]):
    for j in enumerate(cellTypeGeneLists): 
        geneMask = np.array(cellTypeGeneLists[j[1]][:,1], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixCellTypes[i, j[0]] = meanZScore
        # for the sake of calculating cell type probability, we don't want negative z-scores
        if meanZScore < 0:
            cellTypeProbs[i,j[0]] = 0
        else:
            cellTypeProbs[i,j[0]] = scipy_stats.norm.cdf(np.abs(meanZScore))

meanZScoreMatrixInterneurons = np.zeros([sample['geneMatrix'].shape[1], len(cellTypes)])
for i in range(sample['geneMatrix'].shape[1]):
    for j in enumerate(interneuron_information): 
        geneMask = np.array(interneuron_information[j[1]]['geneIdx'], dtype='int32')
        cellTypeMatrix = np.squeeze(np.array(geneMatrixZScore[geneMask, i]))
        # cellTypeMatrix = cellTypeMatrix[geneMask, :]
        # cellTypeMatrixMeanZScore = np.squeeze(np.array(np.mean(cellTypeMatrix, axis=0)))
        meanZScore = np.mean(cellTypeMatrix)
        meanZScoreMatrixInterneurons[i, j[0]] = meanZScore
