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

#%% function for loading gene matrix from allen data

def loadGeneMatrixFromH5ad(h5adLocation):
    # builds a csr matrix from data in the h5ad file
    f = h5py.File(h5adLocation)
    x_attrs = dict(f['X'].attrs)
    gene_matrix = csr_matrix((f['X']['data'],f['X']['indices'],f['X']['indptr']),x_attrs['shape'])
    f.close()
    return gene_matrix

geneMatrix = loadGeneMatrixFromH5ad(fileLocation)
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
tissuePositionList = loadCoordinatesFromH5ad(fileLocation, displayScatter=False)

#%% can use these two combined to plot individual genes
# can change the index in the geneMatrix color call below to change the gene displayed
plt.figure()
plt.scatter(tissuePositionList[:,0], tissuePositionList[:,1], c=np.squeeze(np.array(geneMatrix[:,1].todense())), cmap='Reds')
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