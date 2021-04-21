import torch
import torch.nn as nn
import numpy as np
import pandas as pd

colinears_path = 'data/config/GLOVe/COLINEARITIES.csv'
#colinears_path = 'data/config/RoBERTA/COLINEARITIES.csv'

class mod(nn.Module):

    def __init__(self, config):
        """
        Specific model uses similarity metric of distance divided by feature on
        axis:

                                    ((fa-ua)/fa)^2

        The speaker module is designed to take in a config file that returns the
        following on command:

            (1) config.source() should return all source domains, with an
                an optional parameter to restrict what it returns to only
                certain source domains in the config's list.

            (2) config.axis() should return all possible axes in the config
                file, with an optional parameter to restict the output.

            (3) config.intent() should return all intended paraphrases, adjective
                or otherwise, with an optional parameter to restrict its output.


        As an interesting aside, I did a Grid-Search to find the optimal
        combination of lambda and sigma values. You get .895 acc when you set
        sigma to .5 and lambda to 5. Additionally, the MSE of L0 for this to
        Kao's empirical P(f|c) probabilities is .215--still less than simple
        softmax.

        :param config: the config file. A class with all needed dataframes and
                       arrays
        """
        super(mod, self).__init__()
        self.c = config
        self.soft = nn.Softmax(dim=1)
        # (HAVING USED JAGS TO ESTIMATE PARAMETERS) self.n = torch.distributions.Normal(.796, .977)
        # (USE IF NOT SQUARING EVERYTHING) self.n = torch.distributions.Normal(0, .5)
        self.n = torch.distributions.HalfNormal(.8)


    """
    TRUE RSA
    """
    def prob(self, x):
        return torch.exp(self.n.log_prob(x))

    def l0(self):
        return ((self.c.Ai.unsqueeze(1)-self.c.As)/self.c.Ai.unsqueeze(1))**2

    def u1(self, q):
        L0 = self.prob(self.l0()[:, :, self.c.sel_axis([q])])
        # L0 = 1-self.l0()[:, :, self.c.sel_axis([q])]
        return torch.log(L0)

    def snn(self, u, q, lam=1):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))

        S1 = self.soft(lam*self.u1(q))[self.c.sel_intent(f), self.c.sel_source([u])]
        return S1, f

    #######################################################################
    ############### Some test functions
    #######################################################################
    def snn_TEST(self, u, q, lam=1):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        S1 = self.soft(lam*self.u1(q))[self.c.sel_intent(f)]#[self.c.sel_intent(f), self.c.sel_source([u])]
        median = S1.median(dim=1)[0]
        return (S1[:,self.c.sel_source([u])].squeeze(-1) > median).float(), f

    def just_l0(self,u,q,lam=0):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        return self.prob(self.l0()[self.c.sel_intent(f), self.c.sel_source([u]), self.c.sel_axis([q])]), f

    def just_pct_g(self,u,q,lam=0):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        probs = self.prob(self.l0()[self.c.sel_intent(f), :, self.c.sel_axis([q])])
        uu = probs[:, self.c.sel_source([u])]
        resp = (uu >= probs).float().mean(dim=-1)
        return resp.view(-1), f

    def pad(self, d):
        loc = self.c.c[self.c.sel_context([d])]
        vals = self.c.i @ loc.T
        n = torch.distributions.Normal((loc @ loc.T).view(-1), vals.var())
        return torch.exp(n.log_prob(vals))

class modCOMPLETE(nn.Module):

    def __init__(self, config):
        """
        Specific model uses similarity metric of distance divided by feature on
        axis:

                                    ((fa-ua)/fa)^2

        The speaker module is designed to take in a config file that returns the
        following on command:

            (1) config.source() should return all source domains, with an
                an optional parameter to restrict what it returns to only
                certain source domains in the config's list.

            (2) config.axis() should return all possible axes in the config
                file, with an optional parameter to restict the output.

            (3) config.intent() should return all intended paraphrases, adjective
                or otherwise, with an optional parameter to restrict its output.


        As an interesting aside, I did a Grid-Search to find the optimal
        combination of lambda and sigma values. You get .895 acc when you set
        sigma to .5 and lambda to 5. Additionally, the MSE of L0 for this to
        Kao's empirical P(f|c) probabilities is .215--still less than simple
        softmax.

        :param config: the config file. A class with all needed dataframes and
                       arrays
        """
        super(modCOMPLETE, self).__init__()
        self.c = config
        self.soft = nn.Softmax(dim=1)
        # (HAVING USED JAGS TO ESTIMATE PARAMETERS) self.n = torch.distributions.Normal(.796, .977)
        # (USE IF NOT SQUARING EVERYTHING) self.n = torch.distributions.Normal(0, .5)
        self.n = torch.distributions.HalfNormal(.8)


    """
    TRUE RSA
    """
    def prob(self, x):
        return torch.exp(self.n.log_prob(x))

    def l0(self):
        return ((self.c.Ai.unsqueeze(1)-self.c.As)/self.c.Ai.unsqueeze(1))**2

    def pfc(self):
        return ((self.c.Ai.unsqueeze(1) - self.c.Ac) / self.c.Ai.unsqueeze(1)) ** 2

    def u1(self):
        L0 = self.prob(self.l0())
        return torch.log(L0)

    def snn(self, u, lam=1):
        # S1 = self.u1()[:, self.c.sel_source([u])]
        S1 = self.soft(lam*self.u1())[:, self.c.sel_source([u])]
        return S1

    def lnn(self,u,c,pq,pc,lam=3):
        Sn = self.snn(u,lam=lam) # F x 1 x A
        pfc = self.prob(self.pfc()[:, self.c.sel_topic([u,c])])  # F x 2 x A

        L1 = pc.view(-1,1) * pfc * (pq * Sn) # F x 2 x A
        L1[L1.isnan()] = 0.
        L1 = L1.sum(dim=-1)
        return L1/L1.sum() # P(c) Σ P(f|c) P(a) S1(u|a)

    #######################################################################
    ############### Some test functions
    #######################################################################
    def snn_TEST(self, u, q, lam=1):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        S1 = self.soft(lam*self.u1(q))[self.c.sel_intent(f)]#[self.c.sel_intent(f), self.c.sel_source([u])]
        median = S1.median(dim=1)[0]
        return (S1[:,self.c.sel_source([u])].squeeze(-1) > median).float(), f

    def just_l0(self,u,q,lam=0):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        return self.prob(self.l0()[self.c.sel_intent(f), self.c.sel_source([u]), self.c.sel_axis([q])]), f

    def just_pct_g(self,u,q,lam=0):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        probs = self.prob(self.l0()[self.c.sel_intent(f), :, self.c.sel_axis([q])])
        uu = probs[:, self.c.sel_source([u])]
        resp = (uu >= probs).float().mean(dim=-1)
        return resp.view(-1), f

    def pad(self, d):
        loc = self.c.c[self.c.sel_context([d])]
        vals = self.c.i @ loc.T
        n = torch.distributions.Normal((loc @ loc.T).view(-1), vals.var())
        return torch.exp(n.log_prob(vals))


class modS(nn.Module):

    def __init__(self, config):
        """
        Specific model uses similarity metric of distance divided by feature on
        axis:

                                    ((fa-ua)/fa)^2

        The speaker module is designed to take in a config file that returns the
        following on command:

            (1) config.source() should return all source domains, with an
                an optional parameter to restrict what it returns to only
                certain source domains in the config's list.

            (2) config.axis() should return all possible axes in the config
                file, with an optional parameter to restict the output.

            (3) config.intent() should return all intended paraphrases, adjective
                or otherwise, with an optional parameter to restrict its output.


        :param config: the config file. A class with all needed dataframes and
                       arrays
        """
        super(modS, self).__init__()
        self.c = config
        self.soft = nn.Softmax(dim=1)
        self.gamma=4


    """
    TRUE RSA
    """

    def l0(self):
        return torch.softmax(self.gamma*-((self.c.Ai.unsqueeze(1)-self.c.As)/self.c.Ai.unsqueeze(1))**2,1)

    def u1(self, q):
        L0 = self.l0()[:, :, self.c.sel_axis([q])]
        return torch.log2(L0)

    def snn(self, u, q, lam=1):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        S1 = self.soft(lam*self.u1(q))[self.c.sel_intent(f), self.c.sel_source([u])]
        return S1, f

    def pad(self, d):
        loc = self.c.c[self.c.sel_context([d])]
        vals = self.c.i @ loc.T
        n = torch.distributions.Normal((loc @ loc.T).view(-1), vals.var())
        return torch.exp(n.log_prob(vals))

    def just_l0(self,u,q,lam=0):
        f = self.c.dfcol['colinears'].loc[self.c.dfcol['axis'].isin([q])].values[0].replace('&', ' ').split()
        f = self.c.intent(self.c.sel_intent(f))
        L0 = torch.softmax(self.gamma*-((self.c.Ai.unsqueeze(1)-self.c.As)/self.c.Ai.unsqueeze(1))**2,1)
        return L0[self.c.sel_intent(f), self.c.sel_source([u]), self.c.sel_axis([q])], f