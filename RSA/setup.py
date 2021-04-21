from axesRSA.projection import *


class config():

    def __init__(self, sources=None, axes=None, intents=None, contexts=None, colinears=None, granular=False, LR_scaled=True):
        super(config, self).__init__()
        self.dfs = None
        self.dfax = None
        self.dfi = None
        self.dfcol = None
        self.granular = granular

        self.cos = nn.CosineSimilarity(dim=-1)



        #################################################################
        ### Import data
        #################################################################
        if sources: self.dfs = pd.read_csv(sources)
        else: self.dfs = pd.read_csv('data/config/GLOVe/SOURCE.csv')

        if axes: self.dfax = pd.read_csv(axes)
        else: self.dfax = pd.read_csv('data/config/GLOVe/AXES.csv')

        if intents: self.dfi = pd.read_csv(intents)
        else: self.dfi = pd.read_csv('data/config/GLOVe/INTENTS.csv')

        if colinears: self.dfcol = pd.read_csv(colinears)
        else: self.dfcol = pd.read_csv('data/config/GLOVe/COLINEARITIES-2.csv')

        if contexts: self.dfcon = pd.read_csv(contexts)
        else: self.dfcon = pd.read_csv('data/config/GLOVe/CONTEXTS.csv')

        #################################################################
        ### Remove overlap in any animal terms and intents.
        #################################################################
        self.dfi = self.dfi.loc[~self.dfi['lex'].isin(self.dfs['lex'].unique())]
        #Temporarily remove human markers
        # self.dfT = self.dfs.loc[self.dfs['lex'].isin(['man', 'human', 'woman', 'child', #'swimmer', 'boxer', 'surfer'
        #                                               ])].copy()
        self.dfT = self.dfs.copy()
        self.dfs = self.dfs.loc[~self.dfs['lex'].isin(['man', 'human', 'woman', 'child', 'swimmer', 'boxer', 'surfer'])]


        #################################################################
        ### Set-up vector matrices
        #################################################################
        self.ax = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfax['vec'].values])
        self.s = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfs['vec'].values])
        self.C = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfT['vec'].values])
        self.i = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfi['vec'].values])
        #self.d = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfcon['vec'].values])


        #################################################################
        ### Set-up projections onto axes
        #################################################################
        self.As = vector_power(self.s, self.ax).T
        self.Ac = vector_power(self.C, self.ax).T
        self.Ai = vector_power(self.i, self.ax).T
        #self.si = vector_power(self.s, self.i).T

        #################################################################
        ### Set up colinear boolean search function
        #################################################################
        self.colinears = [i.replace('&', ' ').split() for i in self.dfcol['colinears'].values]
        maxima = max([len(i) for i in self.colinears])
        self.colinears = np.array([i + ['0' for _ in range(len(i), maxima)] for i in self.colinears])
        self.colinears = torch.FloatTensor((self.colinears == self.intent().reshape(-1,1,1)).sum(axis=-1))
        self.Ai = self.Ai * self.colinears

        # if LR_scaled:
        #     scale = []
        #     for axis in self.dfax['axis'].values:
        #         sides = self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0]
        #         sides = [self.sel_intent(i.split()) for i in sides.split('&')]
        #         F = [self.Ai[side, self.sel_axis([axis])].mean() for side in sides]
        #         scale += torch.cat([F[0].view(1,-1),F[1].view(1,-1)]).mean().view(1,-1)#[(F[0] - F[1]).view(1,-1)]
        #     scale = torch.cat(scale, dim=0).view(-1,1)
        #     self.As = (self.As.T/scale).T
        #     self.Ai = (self.Ai.T/scale).T


    """
    Intent
    """
    def intent(self, x=[]):
        if len(x) > 0:
            return self.dfi['lex'].values[x]
        else:
            return self.dfi['lex'].values

    def sel_intent(self, x):
        return (self.intent() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_intent(self, x):
        order = self.intent(self.sel_intent(x)).tolist()
        return [order.index(w) for w in x]



    """
    Source
    """
    def source(self, x=[]):
        if len(x) > 0:
            return self.dfs['lex'].values[x]
        else:
            return self.dfs['lex'].values

    def sel_source(self, x):
        return (self.source() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_source(self, x):
        order = self.source(self.sel_source(x)).tolist()
        return [order.index(w) for w in x]

    """
    Axis
    """
    def axis(self, x=[]):
        if len(x) > 0:
            return self.dfax['axis'].values[x]
        else:
            return self.dfax['axis'].values

    def sel_axis(self, x):
        return (self.axis() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_axis(self, x):
        order = self.axis(self.sel_axis(x)).tolist()
        return [order.index(w) for w in x]

    """
    Context
    """
    def context(self, x=[]):
        if len(x) > 0:
            return self.dfcon['lex'].values[x]
        else:
            return self.dfcon['lex'].values

    def sel_context(self, x):
        return (self.context() == np.array(x).reshape(-1, 1)).sum(axis=0).astype(np.bool)

    """
    Topic
    """
    def topic(self, x=[]):
        if len(x) > 0:
            return self.dfT['lex'].values[x]
        else:
            return self.dfT['lex'].values

    def sel_topic(self, x):
        return (self.topic() == np.array(x).reshape(-1, 1)).sum(axis=0).astype(np.bool)

    """
    reference points
    """

    def rp(self, axis):
        if self.granular:
            return self.sel_intent(self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0].replace('&', ' ').split())
        else:
            return self.sel_intent(axis.split('-'))












class config_spec():

    def __init__(self, sources=None, axes=None, intents=None, contexts=None, colinears=None, granular=False):
        super(config_spec, self).__init__()
        self.dfs = None
        self.dfax = None
        self.dfi = None
        self.dfcol = None
        self.granular = granular

        self.cos = nn.CosineSimilarity(dim=-1)



        #################################################################
        ### Import data
        #################################################################
        if sources: self.dfs = pd.read_csv(sources)
        else: self.dfs = pd.read_csv('data/config/GLOVe/SOURCE.csv')

        if axes: self.dfax = pd.read_csv(axes)
        else: self.dfax = pd.read_csv('data/config/GLOVe/AXES.csv')

        if intents: self.dfi = pd.read_csv(intents)
        else: self.dfi = pd.read_csv('data/config/GLOVe/INTENTS.csv')

        if colinears: self.dfcol = pd.read_csv(intents)
        else: self.dfcol = pd.read_csv('data/config/GLOVe/COLINEARITIES-2.csv')



        #################################################################
        ### Remove overlap in any animal terms and intents.
        #################################################################
        self.dfi = self.dfi.loc[~self.dfi['lex'].isin(self.dfs['lex'].unique())]
        #Temporarily remove human markers
        self.dfs = self.dfs.loc[~self.dfs['lex'].isin(['man', 'human', 'woman', 'child', 'swimmer', 'boxer', 'surfer'])]



        #################################################################
        ### Set-up vector matrices
        #################################################################
        self.ax = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfax['vec'].values])
        self.s = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfs['vec'].values])
        self.i = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfi['vec'].values])



        #################################################################
        ### Set-up projections onto axes
        #################################################################
        self.As = vector_power(self.s, self.ax).T
        self.Ai = vector_power(self.i, self.ax).T
        #self.si = vector_power(self.s, self.i).T

        self.scale = []
        for axis in self.dfax['axis'].values:
            sides = self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0]
            sides = [self.sel_intent(i.split()) for i in sides.split('&')]
            F = [self.Ai[side, self.sel_axis([axis])].mean() for side in sides]
            self.scale += [(F[0] - F[1]).view(1,-1).abs()]
        self.scale = torch.cat(self.scale, dim=0).view(-1)


    """
    Intent
    """
    def intent(self, x=[]):
        if len(x) > 0:
            return self.dfi['lex'].values[x]
        else:
            return self.dfi['lex'].values

    def sel_intent(self, x):
        return (self.intent() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_intent(self, x):
        order = self.intent(self.sel_intent(x)).tolist()
        return [order.index(w) for w in x]



    """
    Source
    """
    def source(self, x=[]):
        if len(x) > 0:
            return self.dfs['lex'].values[x]
        else:
            return self.dfs['lex'].values

    def sel_source(self, x):
        return (self.source() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_source(self, x):
        order = self.source(self.sel_source(x)).tolist()
        return [order.index(w) for w in x]

    """
    Axis
    """
    def axis(self, x=[]):
        if len(x) > 0:
            return self.dfax['axis'].values[x]
        else:
            return self.dfax['axis'].values

    def sel_axis(self, x):
        return (self.axis() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_axis(self, x):
        order = self.axis(self.sel_axis(x)).tolist()
        return [order.index(w) for w in x]

    """
    Context
    """

    def context(self, x=[]):
        if len(x) > 0:
            return self.dfc['lex'].values[x]
        else:
            return self.dfc['lex'].values

    def sel_context(self, x):
        return (self.context() == np.array(x).reshape(-1, 1)).sum(axis=0).astype(np.bool)

    """
    reference points
    """

    def rp(self, axis):
        if self.granular:
            return self.sel_intent(self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0].replace('&', ' ').split())
        else:
            return self.sel_intent(axis.split('-'))

class config_orig():

    def __init__(self, sources=None, axes=None, intents=None, contexts=None):
        self.dfs = None
        self.dfax = None
        self.dfi = None

        self.cos = nn.CosineSimilarity(dim=-1)



        #################################################################
        ### Import data
        #################################################################
        if sources: self.dfs = pd.read_csv(sources)
        else: self.dfs = pd.read_csv('data/config/KAO-ANIMALS.csv')

        if axes: self.dfax = pd.read_csv(axes)
        else: self.dfax = pd.read_csv('data/config/MASTER-AXES.csv')

        if intents: self.dfi = pd.read_csv(intents)
        else: self.dfi = pd.read_csv('data/config/kao-lexical-items-vecs.csv')


        #################################################################
        ### Remove overlap in any animal terms and intents.
        #################################################################
        self.dfi = self.dfi.loc[~self.dfi['lex'].isin(self.dfs['lex'].unique())]
        #Temporarily remove human markers
        #self.dfs = self.dfs.loc[~self.dfs['lex'].isin(['man', 'human', 'woman', 'child', 'swimmer', 'boxer', 'surfer'])]



        #################################################################
        ### Set-up vector matrices
        #################################################################
        self.ax = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfax['vec'].values])
        self.s = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfs['vec'].values])
        self.i = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfi['vec'].values])



        #################################################################
        ### Set-up projections onto axes
        #################################################################
        self.As = vector_power(self.s, self.ax).T
        self.Ai = vector_power(self.i, self.ax).T


    """
    Intent
    """
    def intent(self, x=[]):
        if len(x) > 0:
            return self.dfi['lex'].values[x]
        else:
            return self.dfi['lex'].values

    def sel_intent(self, x):
        return (self.intent() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_intent(self, x):
        order = self.intent(self.sel_intent(x)).tolist()
        return [order.index(w) for w in x]



    """
    Source
    """
    def source(self, x=[]):
        if len(x) > 0:
            return self.dfs['lex'].values[x]
        else:
            return self.dfs['lex'].values

    def sel_source(self, x):
        return (self.source() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_source(self, x):
        order = self.source(self.sel_source(x)).tolist()
        return [order.index(w) for w in x]

    """
    Axis
    """
    def axis(self, x=[]):
        if len(x) > 0:
            return self.dfax['axis'].values[x]
        else:
            return self.dfax['axis'].values

    def sel_axis(self, x):
        return (self.axis() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_axis(self, x):
        order = self.axis(self.sel_axis(x)).tolist()
        return [order.index(w) for w in x]

    """
    Context
    """

    def context(self, x=[]):
        if len(x) > 0:
            return self.dfc['lex'].values[x]
        else:
            return self.dfc['lex'].values

    def sel_context(self, x):
        return (self.context() == np.array(x).reshape(-1, 1)).sum(axis=0).astype(np.bool)


    def reset(self):
        self.As = vector_power(self.s, self.ax).T
        self.Ai = vector_power(self.i, self.ax).T
        self.Ac = vector_power(self.c, self.ax).T

class configj():

    def __init__(self, sources=None, axes=None, intents=None, contexts=None, colinears=None, granular=False, LR_scaled=True):
        super(configj, self).__init__()
        self.dfs = None
        self.dfax = None
        self.dfi = None
        self.dfcol = None
        self.granular = granular

        self.cos = nn.CosineSimilarity(dim=-1)



        #################################################################
        ### Import data
        #################################################################
        if sources: self.dfs = pd.read_csv(sources)
        else: self.dfs = pd.read_csv('data/config/GLOVe/SOURCE.csv')

        if axes: self.dfax = pd.read_csv(axes)
        else: self.dfax = pd.read_csv('data/config/GLOVe/AXES.csv')

        if intents: self.dfi = pd.read_csv(intents)
        else: self.dfi = pd.read_csv('data/config/GLOVe/INTENTS.csv')

        if colinears: self.dfcol = pd.read_csv(colinears)
        else: self.dfcol = pd.read_csv('data/config/GLOVe/COLINEARITIES-2.csv')

        if contexts: self.dfcon = pd.read_csv(contexts)
        else: self.dfcon = pd.read_csv('data/config/GLOVe/CONTEXTS.csv')

        #################################################################
        ### Remove overlap in any animal terms and intents.
        #################################################################
        self.dfi = self.dfi.loc[~self.dfi['lex'].isin(self.dfs['lex'].unique())]
        #Temporarily remove human markers
        self.dfs = self.dfs.loc[~self.dfs['lex'].isin(['man', 'human', 'woman', 'child', 'swimmer', 'boxer', 'surfer'])]

        #################################################################
        ### Set up colinear boolean search function
        #################################################################
        self.colinears = [i.replace('&', ' ').split() for i in self.dfcol['colinears'].values]
        self.colinears = np.array([i+['0' for j in range(len(i),16)] for i in self.colinears])

        #################################################################
        ### Set-up vector matrices
        #################################################################
        #self.ax = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfax['vec'].values])
        self.s = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfs['vec'].values])
        self.i = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfi['vec'].values])
        self.c = torch.FloatTensor([[np.float(i) for i in j.replace('[', '').replace(']', '').split(', ')] for j in self.dfcon['vec'].values])


        #################################################################
        ### Set-up projections onto axes
        #################################################################
        # denom = self.cos(self.i.unsqueeze(1), self.s)
        # denom = denom/denom.max()
        self.sc = torch.sqrt(((self.i.unsqueeze(1)-self.s)**2).sum(dim=-1))
        #self.sc = self.i @ self.s.T
        print(self.s.shape)

        # if LR_scaled:
        #     scale = []
        #     for axis in self.dfax['axis'].values:
        #         sides = self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0]
        #         sides = [self.sel_intent(i.split()) for i in sides.split('&')]
        #         F = [self.Ai[side, self.sel_axis([axis])].mean() for side in sides]
        #         scale += torch.cat([F[0].view(1,-1),F[1].view(1,-1)]).mean().view(1,-1)#[(F[0] - F[1]).view(1,-1)]
        #     scale = torch.cat(scale, dim=0).view(-1,1)
        #     self.As = (self.As.T/scale).T
        #     self.Ai = (self.Ai.T/scale).T


    """
    Intent
    """
    def intent(self, x=[]):
        if len(x) > 0:
            return self.dfi['lex'].values[x]
        else:
            return self.dfi['lex'].values

    def sel_intent(self, x):
        return (self.intent() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_intent(self, x):
        order = self.intent(self.sel_intent(x)).tolist()
        return [order.index(w) for w in x]



    """
    Source
    """
    def source(self, x=[]):
        if len(x) > 0:
            return self.dfs['lex'].values[x]
        else:
            return self.dfs['lex'].values

    def sel_source(self, x):
        return (self.source() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_source(self, x):
        order = self.source(self.sel_source(x)).tolist()
        return [order.index(w) for w in x]

    """
    Axis
    """
    def axis(self, x=[]):
        if len(x) > 0:
            return self.dfax['axis'].values[x]
        else:
            return self.dfax['axis'].values

    def sel_axis(self, x):
        return (self.axis() == np.array(x).reshape(-1,1)).sum(axis=0).astype(np.bool)

    def sort_axis(self, x):
        order = self.axis(self.sel_axis(x)).tolist()
        return [order.index(w) for w in x]

    """
    Context
    """

    def context(self, x=[]):
        if len(x) > 0:
            return self.dfcon['lex'].values[x]
        else:
            return self.dfcon['lex'].values

    def sel_context(self, x):
        return (self.context() == np.array(x).reshape(-1, 1)).sum(axis=0).astype(np.bool)

    """
    reference points
    """

    def rp(self, axis):
        if self.granular:
            return self.sel_intent(self.dfcol['colinears'].loc[self.dfcol['axis'].isin([axis])].values[0].replace('&', ' ').split())
        else:
            return self.sel_intent(axis.split('-'))