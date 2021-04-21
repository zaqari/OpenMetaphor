import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_csv('https://raw.githubusercontent.com/justinek/metaphors/master/Data/FeaturePriorExp/featurePriors-set.csv', index_col=0)

animals = {k:j for j,k in enumerate(df['animal'].unique())}

priors = []

for j in animals:
    x = ([df['normalizedProb'].loc[df['animal'].isin([j]) & df['type'].isin(['animal'])]], [df['normalizedProb'].loc[df['animal'].isin([j]) & df['type'].isin(['person'])]])
    priors.append(x)

priori = np.reshape(priors, [32, 2, 8])

animal_feat = np.array([[1, 1, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [1, 0, 0],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]]).reshape(-1, 3)

an_feat = np.load("kaoetal14/worlds.npy")

def Sn(utterance, weights, animal_feat, lam=3):
    delta = weights @ animal_feat.T
    U = delta * priors

    # softmax decision rule
    S = torch.softmax(torch.FloatTensor(lam*U),dim=0).numpy()
    return S[utterance]

def Ln(utterance, goal_weights, p_animal_human, animal_feat, lam=3):
    u = animals[utterance]
    s = Sn(u, goal_weights, animal_feat, lam)
    L1 = priors[u] * s
    L1 = np.squeeze(p_animal_human.reshape(-1,1,1) * L1,1)
    L1 = L1/L1.sum()
    return L1


def L02():
    return priori, an_feat

def Sn2(utterance, weights, lam=3):
    l0 = L02()
    g = np.array(list(weights.keys()))
    p_g = np.array([weights[i] for i in g])
    delta = (l0[1]==g.reshape(-1,1,1,1))

    U = delta * l0[0][:,0,:].reshape(*l0[0][:,0,:].shape, 1)
    U = U.sum(axis=-1)

    # # softmax decision rule
    S = torch.softmax(torch.log(torch.FloatTensor(lam*U)),dim=1).numpy()
    S[np.isnan(S)] = 1e-9
    return S[:,utterance].sum(axis=-1), p_g

def Ln2(utterance, goal_weights, p_animal_human, lam=3):
    """

    :param utterance:
    :param goal_weights: a dictionary with weights per each goal {"large": .33}
    :param p_animal_human:
    :param animal_feat:
    :param lam:
    :return:
    """
    u = animals[utterance]
    s, p_g = Sn2(u, goal_weights, lam)
    L1 = priori[u] * (s*p_g).sum()
    L1 = L1 * p_animal_human.reshape(-1,1)
    L1 = L1/L1.sum()
    return L1

def LnG(utterance, goal_weights, p_animal_human, lam=3):
    """

    :param utterance:
    :param goal_weights: a dictionary with weights per each goal {"large": .33}
    :param p_animal_human:
    :param animal_feat:
    :param lam:
    :return:
    """
    u = animals[utterance]
    s, p_g = Sn2(u, goal_weights, lam)
    L1 = priori[u].reshape(*priori[u].shape,1) * (s*p_g)
    L1 = L1 * p_animal_human.reshape(-1,1,1)
    #denom = L1.sum(axis=1).reshape(2,1,-1)
    L1 = L1/L1.sum()
    return L1