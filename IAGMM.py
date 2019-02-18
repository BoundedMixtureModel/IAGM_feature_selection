import time
import copy
import mpmath
import numpy as np
from utils import *
from scipy import stats

class Sample:
    """Class for defining a single sample"""
    def __init__(self, mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M):
        self.mu = mu
        self.s_l = s_l
        self.s_r = s_r
        self.pi = np.reshape(pi, (1, -1))
        self.lam = lam
        self.r = r
        self.beta_l = beta_l
        self.beta_r = beta_r
        self.w_l = w_l
        self.w_r = w_r
        self.M = M
        self.alpha = alpha

class Samples:
    """Class for generating a collection of samples"""
    def __init__(self, N, D):
        self.sample = []
        self.N = N
        self.D = D

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self, S):
        return self.sample.append(S)

def infinte_mixutre_model(X, Nsamples=100, Nint=50, anneal=False):
    """
    infinite asymmetric gaussian distribution(AGD) mixture model
    using Gibbs sampling
    input:
        Y : the input datasets
        Nsamples : the number of Gibbs samples
        Nint : the samples used for evaluating the tricky integral
        anneal : perform simple siumulated annealing
    output:
        Samp : the output samples
        Y : the input datasets
    """
    # compute some data derived quantities, N is observations number, D is dimensionality number
    N, D = X.shape
    muy = np.mean(X, axis=0)
    vary = np.zeros(D)
    for k in range(D):
        vary[k] = np.var(X[:, k])

    # initialise a single sample
    Samp = Samples(Nsamples, D)

    c = np.zeros(N)            # initialise the stochastic indicators
    pi = np.zeros(1)           # initialise the weights
    rho = np.zeros((1, D))           # initialise the relevancy
    rho[0] = [0.975, 0.975]

    mu = np.zeros((1, D))      # initialise the means
    mu_irr = np.zeros((1, D))
    s_l = np.zeros((1, D))    # initialise the precisions
    s_l_irr = np.zeros((1, D))
    s_r = np.zeros((1, D))
    s_r_irr = np.zeros((1, D))
    n = np.array([N])  # initialise the occupation numbers

    # set first mu to the mean of all data
    mu[0,:] = muy
    mu_irr[0,:] = muy
    # set first pi to 1, because only one component initially
    pi[0] = 1.0
    z_indicators = np.ones((N, 1, D))

    # draw parameter from prior
    beta_l = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
    beta_r = np.array([np.squeeze(draw_invgamma(0.5, 2)) for d in range(D)])
    w_l = np.array([np.squeeze(draw_gamma(0.5, 2*vary[k])) for k in range(D)])
    w_r = np.array([np.squeeze(draw_gamma(0.5, 2*vary[k])) for k in range(D)])
    s_l[0, :] = np.array([np.squeeze(draw_gamma(beta_l[k]/2, 2/(beta_l[k]*w_l[k]))) for k in range(D)])
    s_l_irr[0, :] = np.array([np.squeeze(draw_gamma(beta_l[k]/2, 2/(beta_l[k]*w_l[k]))) for k in range(D)])
    s_r[0, :] = np.array([np.squeeze(draw_gamma(beta_r[k]/2, 2/(beta_r[k]*w_r[k]))) for k in range(D)])
    s_r_irr[0, :] = np.array([np.squeeze(draw_gamma(beta_r[k]/2, 2/(beta_r[k]*w_r[k]))) for k in range(D)])
    delta_a = np.array([np.squeeze(draw_gamma(2, 0.5)) for k in range(D)])
    delta_b = np.array([np.squeeze(draw_gamma(2, 0.5)) for k in range(D)])
    lam = draw_MVNormal(mean=muy, cov=vary)
    r = np.array([np.squeeze(draw_gamma(0.5, 2/vary[k])) for k in range(D)])
    alpha = 1.0/draw_gamma(0.5, 2.0)

    # set only 1 component, m is the component number
    M = 1
    # define the sample
    S = Sample(mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M)
    # add the sample
    Samp.addsample(S)
    print('{}: initialised parameters'.format(time.asctime()))

    # loop over samples
    iter = 1
    oldpcnt = 0
    while iter < Nsamples:
        # recompute muy and covy
        muy = np.mean(X, axis=0)
        for k in range(D):
            vary[k] = np.var(X[:, k])
        precisiony = 1/vary
        posterior_z = draw_posterior_z(X, pi, rho, mu, s_l, s_r, mu_irr, s_l_irr, s_r_irr, N, M, D)

        # the observations belonged to class j
        Xj = [X[np.where(c==j), :] for j, nj in enumerate(n)]
        j = 0
        # draw mu and mu_irr from posterior
        for x, nj in zip(Xj, n):
            x = x[0]
            # for every dimensionality, compute the posterior distribution of mu_jk
            for k in range(D):
                mu[j, k] = MH_Sampling_posterior_mu_jk(mu[j, k], s_l[j, k], s_r[j, k], mu_irr[j, k], s_l_irr[j, k],s_r_irr[j, k],
                                                       r, lam, posterior_z, j, k, x)
                mu_irr[j, k] = MH_Sampling_posterior_mu_jk(mu_irr[j, k], s_l_irr[j, k],s_r_irr[j, k], mu[j, k], s_l[j, k], s_r[j, k],
                                                           r, lam, 1-posterior_z, j, k, x)
            j += 1

        # draw lambda from posterior
        mu_sum = np.sum(mu, axis=0)
        mu_irr_sum = np.sum(mu_irr, axis=0)
        loc_n = np.zeros(D)
        scale_n = np.zeros(D)
        for k in range(D):
            scale = 1 / (precisiony[k] + 2 * M * r[k])
            scale_n[k] = scale
            loc_n[k] = scale * (muy[k] * precisiony[k] + r[k] * (mu_sum[k] + mu_irr_sum[k]))
        lam = draw_MVNormal(loc_n, scale_n)

        # draw r from posterior
        temp_para_sum = np.zeros(D)
        for k in range(D):
            for muj in mu:
                temp_para_sum[k] += np.outer((muj[k] - lam[k]), np.transpose(muj[k] - lam[k]))
            for muj_irr in mu_irr:
                temp_para_sum[k] += np.outer((muj_irr[k] - lam[k]), np.transpose(muj_irr[k] - lam[k]))
        r = np.array([np.squeeze(draw_gamma((2*M+1)/2, 2/(vary[k] + temp_para_sum[k]))) for k in range(D)])

        # draw alpha from posterior. Because its not standard form, using ARS to sampling
        alpha = draw_alpha(M, N)

        # draw sj from posterior (
        for j, nj in enumerate(n):
            Xj = X[np.where(c == j), :][0]
            for k in range(D):
                s_l[j, k] = MH_Sampling_posterior_sljk(mu[j, k], s_l[j, k], s_r[j, k], mu_irr[j, k], s_l_irr[j, k], s_r_irr[j, k],
                                beta_l, w_l, posterior_z, j, k, Xj)
                s_l_irr[j, k] = MH_Sampling_posterior_sljk(mu_irr[j, k], s_l_irr[j, k], s_r_irr[j, k], mu[j, k], s_l[j, k], s_r[j, k],
                                beta_l, w_l, 1-posterior_z, j, k, Xj)
                s_r[j, k] = MH_Sampling_posterior_srjk(mu[j, k], s_l[j, k], s_r[j, k], mu_irr[j, k], s_l_irr[j, k], s_r_irr[j, k],
                                beta_r, w_r, posterior_z, j, k, Xj)
                s_r_irr[j, k] = MH_Sampling_posterior_sljk(mu_irr[j, k], s_l_irr[j, k], s_r_irr[j, k], mu[j, k], s_l[j, k], s_r[j, k],
                                beta_r, w_r, 1-posterior_z, j, k, Xj)

        # compute the unrepresented probability
        # p_unrep = (alpha / (N - 1.0 + alpha)) * integral_approx(X, lam, r, beta_l, beta_r, w_l, w_r,
        #                                                         delta_a, delta_b)
        p_unrep = (alpha / (N - 1.0 + alpha)) * integral_approx(X, lam, r, beta_l, beta_r, w_l, w_r )
        p_indicators_prior = np.outer(np.ones(M + 1), p_unrep)

        # for the represented components
        for j in range(M):
            # n-i,j : the number of oberservations, excluding Xi, that are associated with component j
            nij = n[j] - (c == j).astype(int)
            idx = np.argwhere(nij > 0)
            idx = idx.reshape(idx.shape[0])
            likelihood_for_associated_data = np.ones(len(idx))
            # for i in range(len(idx)):
            #     for k in range(D):
            #         Y = 0
            #         if X[i, k] < mu[j, k]:
            #             Y += rho[j, k] / (np.power(s_l[j, k], -0.5) + np.power(s_r[j, k], -0.5))* \
            #                         np.exp(- 0.5 * s_l[j, k] * np.power(X[i, k] - mu[j, k], 2))
            #         else:
            #             Y += rho[j, k] / (np.power(s_l[j, k], -0.5) + np.power(s_r[j][k], -0.5))* \
            #                         np.exp(- 0.5 * s_r[j, k] * np.power(X[i, k] - mu[j, k], 2))
            #         if X[i, k] < mu_irr[j, k]:
            #             Y += (1 - rho[j, k]) / (np.power(s_l_irr[j, k], -0.5) + np.power(s_r_irr[j, k], -0.5)) \
            #                       * np.exp(- 0.5 * s_l_irr[j, k] * (X[i, k] - mu_irr[j, k]) ** 2)
            #         else:
            #             Y += (1 - rho[j, k]) / (np.power(s_l_irr[j, k], -0.5) + np.power(s_r_irr[j, k], -0.5)) \
            #                       * np.exp(- 0.5 * s_r_irr[j, k] * (X[i, k] - mu_irr[j, k]) ** 2)
            #         likelihood_for_associated_data[i] *= Y
            # p_indicators_prior[j, idx] = nij[idx] / (N - 1.0 + alpha) * likelihood_for_associated_data
            likelihood_for_associated_data = np.ones(len(idx))
            for i in range(len(idx)):
                for k in range(D):
                    if X[i][k] < mu[j][k]:
                        likelihood_for_associated_data[i] *= 1 / (np.power(s_l[j][k], -0.5) + np.power(s_r[j][k], -0.5))* \
                                    np.exp(- 0.5 * s_l[j][k] * np.power(X[i][k] - mu[j][k], 2))
                    else:
                        likelihood_for_associated_data[i] *= 1 / (np.power(s_l[j][k], -0.5) + np.power(s_r[j][k], -0.5))* \
                                    np.exp(- 0.5 * s_r[j][k] * np.power(X[i][k] - mu[j][k], 2))
            p_indicators_prior[j, idx] = nij[idx]/(N - 1.0 + alpha)*likelihood_for_associated_data
        # for j in range(p_indicators_prior.shape[0]):
        #     p_indicators_prior[j, np.where(p_indicators_prior[j] < 0.00001)] = 0.000001
        #
        # print("__")
        # print(p_indicators_prior)
        # for j in range(p_indicators_prior.shape[0]):
        #     if j == 0:
        #         pass
        #     else:
        #         print(np.where(p_indicators_prior[j]>0.00001))
        # print("__")
        # stochastic indicator (we could have a new component)
        c = np.hstack(draw_indicator(p_indicators_prior))

        # draw w from posterior
        w_l = np.array([np.squeeze(draw_gamma(0.5 *(2*M*beta_l[k]+1),\
                        2/(vary[k] + beta_l[k]*np.sum(s_l, axis=0)[k] + beta_l[k]*np.sum(s_l_irr, axis=0)[k]
                        ))) for k in range(D)])
        w_r = np.array([np.squeeze(draw_gamma(0.5 *(2*M*beta_r[k]+1),\
                        2/(vary[k] + beta_r[k]*np.sum(s_r, axis=0)[k] + beta_r[k]*np.sum(s_r_irr, axis=0)[k]
                        ))) for k in range(D)])

        # draw beta from posterior. Because its not standard form, using ARS to sampling.
        beta_l = np.array([draw_beta_ars(w_l, s_l, s_l_irr, M, k)[0] for k in range(D)])
        beta_r = np.array([draw_beta_ars(w_l, s_r, s_r_irr, M, k)[0] for k in range(D)])

        # draw z from Bernoulli distribution
        for i in range(N):
            for j in range(M):
                for k in range(D):
                    z_indicators[i, j, k] = draw_Bernoulli(posterior_z[i, j, k])
        f = np.array([[np.count_nonzero(z_indicators[:, j, k] == 1) for k in range(D)] for j in range(M)])

        # draw rho from posterior
        for j in range(M):
            for k in range(D):
                rho[j, k] = draw_Beta_dist(f[j, k] + delta_a[k], N - f[j, k] + delta_b[k])

        # draw delta from posterior
        for k in range(D):
            delta_a[k] = MH_Sampling_posterior_delta_a(delta_a[k], delta_b[k], rho, k, M)
            delta_b[k] = MH_Sampling_posterior_delta_b(delta_a[k], delta_b[k], rho, k, M)

        # sort out based on new stochastic indicators
        nij = np.sum(c == M)        # see if the *new* component has occupancy
        if nij > 0:
            # draw from priors and increment M
            # newmu = np.array([np.squeeze(draw_normal(lam[k], 1 / r[k])) for k in range(D)])
            newmu = draw_MVNormal(lam, 1/r)
            news_l = np.array([np.squeeze(draw_gamma(beta_l[k] / 2, 2 / (beta_l[k] * w_l[k]))) for k in range(D)])
            news_r = np.array([np.squeeze(draw_gamma(beta_r[k] / 2, 2 / (beta_r[k] * w_r[k]))) for k in range(D)])
            newmu_irr = np.array([np.squeeze(draw_normal(lam[k], 1 / r[k])) for k in range(D)])
            news_l_irr = np.array([np.squeeze(draw_gamma(beta_l[k] / 2, 2 / (beta_l[k] * w_l[k]))) for k in range(D)])
            news_r_irr = np.array([np.squeeze(draw_gamma(beta_r[k] / 2, 2 / (beta_r[k] * w_r[k]))) for k in range(D)])
            # newrho = draw_Beta_dist(delta_a, delta_b)
            newrho = np.array([0.975, 0.975])
            new_z_indicators = np.ones((N, 1, D))

            mu = np.concatenate((mu, np.reshape(newmu, (1, D))))
            s_l = np.concatenate((s_l, np.reshape(news_l, (1, D))))
            s_r = np.concatenate((s_r, np.reshape(news_r, (1, D))))
            mu_irr = np.concatenate((mu_irr, np.reshape(newmu_irr, (1, D))))
            s_l_irr = np.concatenate((s_l_irr, np.reshape(news_l_irr, (1, D))))
            s_r_irr = np.concatenate((s_r_irr, np.reshape(news_r_irr, (1, D))))
            rho = np.concatenate((rho, np.reshape(newrho, (1, D))))
            z_indicators = np.concatenate((z_indicators, new_z_indicators), axis=1)
            M = M + 1
        # find the associated number for every components
        n = np.array([np.sum(c == j) for j in range(M)])

        # find unrepresented components
        badidx = np.argwhere(n == 0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad > 0:
            mu = np.delete(mu, badidx, axis=0)
            s_l = np.delete(s_l, badidx, axis=0)
            s_r = np.delete(s_r, badidx, axis=0)
            rho = np.delete(rho, badidx, axis=0)
            mu_irr = np.delete(mu_irr, badidx, axis=0)
            s_l_irr = np.delete(s_l_irr, badidx, axis=0)
            s_r_irr = np.delete(s_r_irr, badidx, axis=0)
            z_indicators = np.delete(z_indicators, badidx, axis=1)

            # if the unrepresented compont removed is in the middle, make the sequential component indicators change
            for cnt, i in enumerate(badidx):
                idx = np.argwhere(c >= (i - cnt))
                c[idx] = c[idx] - 1
            M -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c == j) for j in range(M)])

        # recompute pi
        pi = n.astype(float)/np.sum(n)

        pcnt = int(100.0 * iter / float(Nsamples))
        if pcnt > oldpcnt:
            print('{}: %--- {}% complete ----------------------%'.format(time.asctime(), pcnt))
            oldpcnt = pcnt

        # add sample
        S = Sample(mu, s_l, s_r, pi, lam, r, beta_l, beta_r, w_l, w_r, alpha, M)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        iter += 1
        print(n)
        print(mu)
        print(s_l)
        print(s_r)
        print(rho)
    return Samp, X, c, n

