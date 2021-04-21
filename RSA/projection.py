import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA as PCA
import pandas as pd
import numpy as np

def vecs_from_df(df, identifier, vector):
    return {df[identifier].loc[i]: torch.FloatTensor([np.float(j) for j in df[vector].loc[i].replace('[', '').replace(']','').split(', ')]).view(1,-1) for i in df.index}

def vector_projection(nns, axis):
    num = nns @ axis.T
    denom = (axis * axis).sum(axis=-1).unsqueeze(-1)
    projection = ((num/denom) * axis)
    return projection

"""def vector_power(nns, axis):
    num = nns @ axis.T
    denom = (axis @ axis.T).unsqueeze(1)
    return (num / denom)"""

def vector_power(nns, axis):
    num = nns @ axis.T
    denom = torch.sqrt((axis**2).sum(axis=-1).unsqueeze(-1))
    #denom = (axis**2).sum(axis=-1).unsqueeze(-1)
    return (num.T / denom)



def pca(X, PCAn=3, batch_size=3):
    p = PCA(n_components=PCAn, batch_size=batch_size)
    #y = None
    #for _ in range(epochs-1):
    #    y = p.fit(X.numpy())
    y = p.fit_transform(X.numpy())
    return pd.DataFrame(y, columns=[str(i) for i in range(PCAn)])


def scatter3D(cols,data, labels=None, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = data[cols[0]], data[cols[1]], data[cols[2]]
    ax.scatter(xs, ys, zs, s=100, alpha=.6, edgecolors='w')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])

    if bool(labels):
        for i in data.index:
            ax.text(data[cols[0]].loc[i], data[cols[1]].loc[i], data[cols[2]].loc[i], data[labels].loc[i])

    plt.show()

def line3D(cols, data, labels=None, title='3D line plot', lims=(1,1)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xs, ys, zs = data[cols[0]], data[cols[1]], data[cols[2]]
    ax.scatter(xs,ys,zs, marker='o')
    #ax.plot(xs, ys, zs, label=title)

    if bool(labels):
        for i in data.index:
            ax.text(data[cols[0]].loc[i], data[cols[1]].loc[i], data[cols[2]].loc[i], data[labels].loc[i])

    plt.xlim(-lims[0], lims[0])
    plt.ylim(-lims[1], lims[1])
    plt.show()


def scatter3Dx2(cols, data1, data2, labels=None, title='3D line plot', lims=(1,1)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xs, ys, zs = data1[cols[0]], data1[cols[1]], data1[cols[2]]
    ax.scatter(xs,ys,zs, marker='o')
    ax.plot(xs, ys, zs, label=title)
    if bool(labels):
         for i in data1.index:
             ax.text(data1[cols[0]].loc[i], data1[cols[1]].loc[i], data1[cols[2]].loc[i], data1[labels].loc[i])

    xs, ys, zs = data2[cols[0]], data2[cols[1]], data2[cols[2]]
    ax.scatter(xs, ys, zs, marker='o')
    ax.plot(xs, ys, zs, label=title)
    if bool(labels):
         for i in data2.index:
             ax.text(data2[cols[0]].loc[i], data2[cols[1]].loc[i], data2[cols[2]].loc[i], data2[labels].loc[i])

    plt.xlim(-lims[0], lims[0])
    plt.ylim(-lims[1], lims[1])
    plt.show()


def scatter2Dx2(cols, data1, data2, labels=None, title='3D line plot', lims=(1,1)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    xs, ys = data1[cols[0]], data1[cols[1]]
    ax.scatter(xs,ys, marker='o')
    ax.plot(xs, ys, label=title)
    if bool(labels):
         for i in data1.index:
             ax.text(data1[cols[0]].loc[i], data1[cols[1]].loc[i], data1[labels].loc[i])

    xs, ys = data2[cols[0]], data2[cols[1]]
    ax.scatter(xs, ys, marker='o')
    ax.plot(xs, ys, label=title)
    if bool(labels):
         for i in data2.index:
             ax.text(data2[cols[0]].loc[i], data2[cols[1]].loc[i], data2[labels].loc[i])

    plt.xlim(-lims[0], lims[0])
    plt.ylim(-lims[1], lims[1])
    plt.show()

def scatter2D(cols,data, labels=None, figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.scatter(data[cols[0]], data[cols[1]], s=100, alpha=.6, edgecolors='w')

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])

    if bool(labels):
        for i in data.index:
            ax.text(data[cols[0]].loc[i], data[cols[1]].loc[i], data[labels].loc[i])

    plt.show()


def line2D(cols, data, labels=None, title='3D line plot'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs, ys = data[cols[0]], data[cols[1]]
    ax.scatter(xs,ys, marker='o')
    ax.plot(xs, ys, label=title)

    if bool(labels):
        for i in data.index:
            ax.text(data[cols[0]].loc[i], data[cols[1]].loc[i], data[labels].loc[i])

    plt.show()