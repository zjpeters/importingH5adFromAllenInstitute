#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:44:06 2025

@author: zjpeters
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import anndata

"""
data is stored in 'h5ad' file type
This is an AnnData file type, with the specification listed here:
    https://anndata.readthedocs.io/en/latest/fileformat-prose.html
"""

#%% load registered h5ad file and print keys
fileLocation = os.path.join('/','media','zjpeters','Expansion','merscopeDataFromAllenInstitute','sourcedata','mouse_609882_registered_082725.h5ad')

#%% extract barcodes from data
f = h5py.File(fileLocation)
print(f['obs'].keys())
y = len(np.unique(f['obs']['z_CCF'][:]))
# y = (f['obs']['brain_section_barcode']['codes'][:])
# print(f['X']['data'])
# print(f['obsm'].keys())
# print(f['obsm']['spatial'])
# x = dict(f['X'].attrs)
f.close()
#%% function for loading gene matrix from allen data
"""
can find the slice barcodes at:
    f['obs']['brain_section_barcode']['categories'][:]
can find the associations of cells to slices at:
    f['obs']['brain_section_barcode']['codes'][:]
"""
def loadGeneMatrixFromH5ad(h5adLocation):
    # builds a csr matrix from data in the h5ad file
    f = h5py.File(h5adLocation)
    x_attrs = dict(f['X'].attrs)
    gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
    gene_list = np.array(f['var']['_index'])
    f.close()
    return gene_matrix, gene_list


#%% function for loading tissue position coordinates from h5ad

def loadCoordinatesFromH5ad(h5adLocation, displayScatter=True):
    # loads the coordinates from h5ad as [N,2] numpy array
    # display scatter just displays all coordinates, no expression levels
    f = h5py.File(h5adLocation)
    tissue_coordinates = f['obsm']['spatial'][:,0:2]
    if displayScatter == True:
        plt.figure()
        plt.scatter(f['obsm']['spatial'][:,0], f['obsm']['spatial'][:,1])
        plt.show()
    f.close()
    return tissue_coordinates

#%% combine loading of matrix and coordinates for single slice

def loadSingleSliceFromH5ad(h5adLocation, sliceNumber, displayScatter=False):
    f = h5py.File(h5adLocation)
    x_attrs = dict(f['X'].attrs)
    gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
    gene_list = np.array(f['var']['_index'])
    f = h5py.File(h5adLocation)
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
    sample['geneList'] = gene_list
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
sample = loadSingleSliceFromH5ad(fileLocation, 10, displayScatter=True)


#%% plot ccf coordinates
plt.close('all')
fig,ax = plt.subplots(1,1)
ax.scatter(sample['ccfCoordinates'][:,2], sample['ccfCoordinates'][:,1])
ax.yaxis.set_inverted(True)
plt.show()
#%% plot multiple sequential images

sample10 = loadSingleSliceFromH5ad(fileLocation, 10, displayScatter=True)
sample11 = loadSingleSliceFromH5ad(fileLocation, 11, displayScatter=True)
#%%
plt.close('all')
fig,ax = plt.subplots(1,1)
ax.scatter(sample10['ccfCoordinates'][:,2], sample10['ccfCoordinates'][:,1])
ax.scatter(sample11['ccfCoordinates'][:,2], sample11['ccfCoordinates'][:,1], alpha=0.5)
ax.yaxis.set_inverted(True)
plt.show()
#%% can use these two combined to plot individual genes
# can change the index in the geneMatrix color call below to change the gene displayed
plt.figure()
plt.scatter(tissuePositionList[:,0], tissuePositionList[:,1], c=np.squeeze(np.array(gene_matrix[:,360].todense())), cmap='Reds', s=3)
plt.axis('equal')
plt.show()
#%% messy code used to figure out data within the h5ad file

f = h5py.File(fileLocation)
print(len(f['obs'].keys()))
# print(f['X']['data'])
# print(f['obs'].keys())
# print(f['obsm']['spatial'])
x = dict(f['X'].attrs)
# y = csr_matrix()
# print(x)
# f['X'].visititems(print)
# print(f.keys())
# can display scatter of spatial points
geneMatrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']), x['shape'])
tissuePositionList = f['obsm']['spatial'][:,0:2]
# plt.scatter(f['obsm']['spatial'][:,0], f['obsm']['spatial'][:,1], c=np.squeeze(np.array(geneMatrix[:,1].todense())))
f.close()

#%% plot data

geneExpressionVector = np.squeeze(np.array(geneMatrix[:,0].todense()))
geneExpressionNoZerosIdx = geneExpressionVector > 0
geneExpressionNoZeros = geneExpressionVector[geneExpressionNoZerosIdx]
tissuePositionsNoZeros = tissuePositionList[geneExpressionNoZerosIdx,:]
plt.close('all')
plt.scatter(tissuePositionsNoZeros[:,0], tissuePositionsNoZeros[:,1], c=geneExpressionNoZeros, cmap='Reds')
plt.show()

#%% tried using the anndata function for loading file but overloads memory
adata = anndata.read_h5ad(fileLocation, backed='r')
print(adata)