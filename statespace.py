"""
State space analysis for decision tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import time
import copy
from collections import OrderedDict
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
from task import *
from run import Run

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


save_addon = 'tf_latest_500'
rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]

def f_sub_mean(x):
    # subtract mean activity across batch conditions
    assert(len(x.shape)==3)
    x_mean = x.mean(axis=1)
    for i in range(x.shape[1]):
        x[:,i,:] -= x_mean
    return x

sub_mean=True
print('Starting standard analysis of the CHOICEATTEND task...')
with Run(save_addon) as R:

    n_tar = 6
    batch_size = n_tar**2
    batch_shape = (n_tar,n_tar)
    ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    tar1_loc  = 0
    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

    tar_str_range = 0.2
    tar1_mod1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar_mod1/(n_tar-1)
    tar2_mod1_strengths = 2 - tar1_mod1_strengths
    tar1_mod2_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar_mod2/(n_tar-1)
    tar2_mod2_strengths = 2 - tar1_mod2_strengths

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_mod1_strengths' : tar1_mod1_strengths,
              'tar2_mod1_strengths' : tar2_mod1_strengths,
              'tar1_mod2_strengths' : tar1_mod2_strengths,
              'tar2_mod2_strengths' : tar2_mod2_strengths,
              'tar_time'    : 800}

    h_samples = dict()
    for rule in rules:
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        # Only study target epoch
        epoch = task.epochs['tar1']
        h_sample = R.f_h(task.x)[epoch[0]:epoch[1],...][::20,...]
        if sub_mean:
            h_sample = f_sub_mean(h_sample)
        h_samples[rule] = h_sample


def show3D(h_tran, separate_by, pcs=(0,1,2), **kwargs):
    if separate_by == 'tar1_mod1_strengths':
        separate_bys = tar1_mod1_strengths
        colors = sns.color_palette("RdBu_r", len(np.unique(separate_bys)))
    elif separate_by == 'tar1_mod2_strengths':
        separate_bys = tar1_mod2_strengths
        colors = sns.color_palette("BrBG", len(np.unique(separate_bys)))
    else:
        raise ValueError

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(np.unique(separate_bys)):
        h_plot = h_tran[:,separate_bys==s,:]
        for j in range(h_plot.shape[1]):
            ax.plot(h_plot[:,j,pcs[0]], h_plot[:,j,pcs[1]], h_plot[:,j,pcs[2]],
                    '.-', markersize=2, color=colors[i])
    if 'azim' in kwargs:
        ax.azim = kwargs['azim']
    if 'elev' in kwargs:
        ax.elev = kwargs['elev']
    ax.elev = 62
    plt.show()

def test_PCA_ring():
    # Test PCA with simple ring representation
    n_t = 3
    n_loc = 256
    n_ring = 64
    h_simple = np.zeros((n_t, n_loc, n_ring))

    pref = np.arange(0,2*np.pi,2*np.pi/n_ring) # preferences
    locs = np.arange(n_loc)/n_loc*2*np.pi
    ts   = np.arange(n_t)/n_t
    for i, loc in enumerate(locs):
        dist = get_dist(loc-pref) # periodic boundary
        dist /= np.pi/8
        h_simple[:,i,:] = 0.8*np.exp(-dist**2/2)
        h_simple[:,i,:] = (h_simple[:,i,:].T * ts).T

    from sklearn.decomposition import PCA
    pca = PCA()
    h_tran = pca.fit_transform(h_simple.reshape((-1, n_ring))).reshape((n_t, n_loc, n_ring))
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    colors = sns.color_palette("husl", n_loc)

    h_tran += np.random.randn(*h_tran.shape)*0.004

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([.2,.2,.7,.7])
    for i in range(n_loc):
        ax.plot(h_tran[:,i,0], h_tran[:,i,1], '-', linewidth=0.3, color=colors[i])
    ax.set_aspect('equal')
    plt.savefig('figure/temp.pdf')
    plt.show()

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([.2,.2,.7,.7])
    ax.plot(evr, 'o-')
    plt.show()


# nt, nb, nh = h_samples[CHOICEATTEND_MOD1].shape
# from sklearn.decomposition import PCA
# pca = PCA()
# #pca.fit(h_samples[CHOICEATTEND_MOD1].reshape((-1, nh)))
# pca.fit(np.concatenate((h_samples[CHOICEATTEND_MOD1].reshape((-1, nh)),h_samples[CHOICEATTEND_MOD2].reshape((-1, nh))), axis=0))
#
# h_trans = dict()
# for rule in rules:
#     h_trans[rule] = pca.transform(h_samples[rule].reshape((-1, nh)))

#h_tran = h_trans[CHOICEATTEND_MOD1].reshape((nt, nb, -1))
#show3D(h_tran, 'tar1_mod1_strengths', azim=-62, elev=62)
#show3D(h_tran, 'tar1_mod2_strengths', azim=-62, elev=62)

#h_tran = h_trans[CHOICEATTEND_MOD2].reshape((nt, nb, -1))
#show3D(h_tran, 'tar1_mod1_strengths', azim=-62, elev=62)
#show3D(h_tran, 'tar1_mod2_strengths', azim=-62, elev=62)



# Regression
rule = CHOICEATTEND_MOD1
h = h_samples[rule]

from sklearn import linear_model
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
X = np.array([tar1_mod1_strengths, tar1_mod2_strengths]).T
regr.fit(X, h[-1,:,0])
h1 = regr.predict(X)

plt.plot(h1[:], h[-1,:,0], 'o')
plt.show()