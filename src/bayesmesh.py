from __future__ import division

import numpy as np
from scipy.stats import expon, geom, multivariate_normal as mvn, dirichlet, norm

from cvxpy import Variable, Problem, Parameter, sum_squares, Minimize
import matplotlib.pyplot as plt

from complex import Vertex, Simplex, SimplicialComplex

import time

eps = 1e-8

class Obs(object):
    
    """
    latent variables for latent variable GP model

    eventually, these will generate the actual observations
    """
    
    def __init__(self, pt, cmplx, sigma=.05, s_source=None):
        """
        pt: position of the point
        s_source: pointer to simplex that generated point
        """
        self.pt = pt
        self.cmplx = cmplx
        self.d = pt.shape[0]
        # Simplex this observation is from
        self.s = s_source
        self.noise_dist = norm(loc=0, scale=sigma)
        if self.s is None:
            self.dist, self.q, self.s = cmplx.proj(self.pt)
        else:
            self.dist, self.q = self.s.proj(self.pt)
        assert self.q is not None
            
    # @profile
    def log_likelihood(self):
        return self.noise_dist.logpdf(self.dist)

    def set_source(self, s):
        self.s = s
        self.dist, self.q = self.s.proj(self.pt)

    def draw(self, ax):
        ax.scatter(self.pt[0], self.pt[1], marker='o')
        ((x_start, y_start), (x_end, y_end)) = (
            self.pt, self.q)
        X = [x_start, x_end]
        Y = [y_start, y_end]
        ax.scatter(float(self.q[0]), float(self.q[1]), marker='v')
        ax.plot(X, Y, ls='--')
        
        

class BayesMesh1D(object):

    """
    wrapper around a set of observations and a simplicial complex

    places a generic prior on complexes
    """
    def __init__(self, obs_pts=None, cmplx=None, gamma=.2, lmbda=.5, obs_sigma=.05, propose_sigma=.1,
                 d=2, obs=None, N=None, P=None):
        """
        gamma: geometric variable for prior on number of simplices
        sigma_sq: variance of 
        d: dimension of embedding space
        """

        assert not (obs_pts is None and cmplx is None)

        self.gamma = gamma
        self.N_prior = geom(gamma)

        self.d = d
        self.lmbda = lmbda
        self.len_prior = expon(self.lmbda)

        self.propose_mvn = mvn(np.zeros(self.d), propose_sigma*np.eye(self.d))
        self.obs_dist = mvn(np.zeros(self.d), obs_sigma*np.eye(self.d))

        self.cmplx = cmplx
        if self.cmplx is None:
            # obs_pts is not None
            self.cmplx = SimplicialComplex()
            ## this is a 1d complex
            self.cmplx.initialize(obs_pts, 1)

        self.N = self.cmplx.simplex_count()

        if obs_pts is None:
#            self.sample_obs(self.N * 10)
            self.sample_obs(self.N * 100)
        else:
            self.observations = []                        
            for pt in obs_pts:
                self.observations.append(Obs(pt, self.cmplx))
            
    def prior_s_ll(self, s):
        s_len = s.vertices[0].dist(s.vertices[1])
        ## add 1 b/c expon is a dist over [1, \infty)        
        ## and length can be [0, \infty)
        return self.len_prior.logpdf(s_len + 1)

    def prior_N_ll(self):
        return self.N_prior.logpmf(self.N)

    def prior_ll(self):
        # print 'simplex size prior', np.sum(self.prior_s_ll(s) for s in self.cmplx.simplices)
        # print 'N prior ll', self.prior_N_ll()
        return (np.sum(self.prior_s_ll(s) for s in self.cmplx.simplices) +
                self.prior_N_ll())

    def sample_obs(self, n_samples):
        self.observations = []
        pts = np.zeros((n_samples, self.d))
        for i in range(n_samples):
            s = np.random.choice(self.cmplx.simplices)
            lmbda = np.random.rand()
            pt_src = lmbda * s.vertices[0].v + (1-lmbda) * s.vertices[1].v
            pt = self.obs_dist.rvs() + pt_src
            pts[i, :] = pt
            self.observations.append(Obs(pt, self.cmplx))
        return pts

    def set_obs(self, pts, draw=False):
        self.observations = []
        n_obs = pts.shape[1]
        for i in range(n_obs):
            pt = pts[:, i].reshape((self.d, 1))
            o_i = Obs(pt, self.cmplx)
            self.observations.append(o_i)
        if draw:
            self.draw(block=True)

    def obs_ll(self):
        ll = 0
        for o in self.observations:
            ## need to compute distance to observation
            ll += o.log_likelihood()
        return ll

    # @profile
    def log_likelihood(self):
        prior_ll = self.prior_ll()
        obs_ll = self.obs_ll()
        # print 'prior_ll', prior_ll, 'obs_ll', obs_ll
        return prior_ll + obs_ll

    # @profile
    def mh(self, samples=500, draw=False, gt_ll=None):
        if draw:
            fig, axarr = plt.subplots(2)
            axarr[1].set_title('Log-Likelihood')
            log_likelihoods = [self.log_likelihood()]
            l_mcmc,  = axarr[1].plot(range(len(log_likelihoods)), log_likelihoods, label='MCMC')
            if gt_ll:
                gt_ll_arr = np.ones(samples+1)*gt_ll
                l_gt,  = axarr[1].plot(range(samples+1), gt_ll_arr, label='Ground Truth')

            plt.legend(loc='best')
            plt.show(block=False)
            plt.draw()
            raw_input('go?')
                
        proposals = [self.propose_vertices, self.propose_correspondence]
        proposal_p = [.4, .6]
        accept = 0
        print self.log_likelihood()
        for i in range(samples):
            propose_i = np.random.choice(proposals, p=proposal_p)

            # for s in self.cmplx.simplices:
            #     print s.vertices

            f_apply, f_undo = propose_i()
            accept_i = mh_step(self, f_apply, f_undo, verbose=False)
            accept += accept_i
            # print 'accepted', accept_i, self.log_likelihood()
            # if np.isnan(self.log_likelihood()):
            #     import pdb; pdb.set_trace()
            log_likelihoods.append(self.log_likelihood())
            if draw and i%draw == 0:
                print "accept rate", accept / (i+1)                
                l_mcmc.set_data(range(len(log_likelihoods)), log_likelihoods)
                self.draw(block=False, ax=axarr[0])
                plt.draw()
                time.sleep(.1)

        if draw:
            l_mcmc.set_data(range(len(log_likelihoods)), log_likelihoods)
            self.draw(block=True, ax=axarr[0])
            
        
    # @profile            
    def propose_vertices(self):
        ## symmetric
        s = np.random.choice(self.cmplx.simplices)
        v_idx = np.random.choice([0, 1])
        # print "v_idx", v_idx
        v_old = s.vertices[v_idx, :].copy()
        # print "v_old", v_old
        offset = self.propose_mvn.rvs()
        v_new = v_old + offset
        # print "v_new", v_new

        def f_apply():           
            # print 'before apply'
            # self.print_vertices()
            s.set_vertex(v_idx, v_new)
            self.check_pointers()
            # print 'after apply'
            # self.print_vertices()
            ## proposal probabilities don't matter here
            return 0, 0

        def f_undo():
            # print "v_old", v_old
            # print 'before undo'
            # self.print_vertices()
            s.set_vertex(v_idx, v_old)
            self.check_pointers()
            # print 'after undo'
            # self.print_vertices()
            return 

        return (f_apply, f_undo)

    # @profile
    def propose_correspondence(self):
        ## project random point near an obs onto Mesh

        o = np.random.choice(self.observations)
        lmbda_old = o.lmbda
        s_old = o.s

        offset = self.propose_mvn.rvs().reshape((self.d, 1))
        pt_new = o.pt + offset
        
        s_new, lmbda_new = self.proj(pt_new)

        def f_apply():
            o.s = s_new
            o.lmbda = lmbda_new
            return (0, 0)

        def f_undo():
            o.s = s_old
            o.lmbda = lmbda_old
            return

        return (f_apply, f_undo)

    # def propose_simplex_birth(self):
    #     s_idx = np.random.randint(self.N)
    #     s = self.simplices[s_idx]
    #     v_mid = (s.vertices[0] + s.vertices[1]) / 2
    #     offset = self.propose_mvn.rvs()
    #     v_new = v_mid + offset

    #     ll_apply = (self.propose_mvn.logpdf(offset) +
    #                 np.log(1/self.N))

    #     ll_reverse = np.log(1/(self.N + 1))
        
    #     def f_apply():
    #         s_new = Simplex(v_new, s.vertices[1])
    #         self.splice(s, s_new, 1)
    #         s_new.list_idx = self.N
    #         self.N += 1
    #         return ll_apply, ll_reverse

    #     def f_undo():
    #         s_new = s.neighbors[1][0]
    #         self.remove(s_new)
    #         self.N -= 1
    #         return
        
    #     return f_apply, f_undo

    # def propose_simplex_death(self):
    #     if self.N == 1:
    #         # can't remove only simplex
    #         f_apply = lambda : (0, 0)
    #         f_undo = lambda: None
    #         return f_apply, f_undo
    #     s_idx = np.random.randint(self.N)
    #     s = self.cmplx.simplices[s_idx]

    #     # which vertex gets deleted
    #     rm_ind = np.random.choice([0, 1])
    #     if s.neighbors[rm_ind] is None:
    #         # if neighbors doesn't exist swap
    #         rm_ind = 1-rm_ind

    #     v_rm = s.vertices[rm_ind]
    #     s_adj = s.neighbors[rm_ind][0]
        
    #     v_keep = s.vertices[1-rm_ind]
    #     v_connect = s_adj.vertices[rm_ind]

    #     v_mid = (v_keep + v_connect) / 2
    #     offset = v_rm - v_mid 

    #     ll_reverse = (self.propose_mvn.logpdf(offset) +
    #                 np.log(1/(self.N-1)))

    #     ll_apply = np.log(1/(self.N))

    #     def f_apply():
    #         self.remove(s)

    #     pass

    def draw(self, ax=None, block=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        for s in self.cmplx.simplices:
            ((x0, y0), (x1, y1)) = s.vertices[0].v, s.vertices[1].v
            ax.plot([x0, x1], [y0, y1])
        if self.observations:
            for o in self.observations:
                o.draw(ax)
        plt.show(block=block)

# @profile
def mh_step(model, f_apply, f_undo, verbose=False):
    ll_old = model.log_likelihood()
    ll_propose, ll_reverse = f_apply()
    ll_new = model.log_likelihood()
    
    ll_alpha = min(0, (ll_new + ll_reverse) - (ll_old + ll_propose))    
    if verbose:
        print 'll_old', ll_old, 'll_new', ll_new, 'acceptance probability', np.exp(ll_alpha)
    
    if np.random.rand() > np.exp(ll_alpha):
        # reject proposal
        if verbose:
            print 'reject'
        f_undo()
        return 0
    return 1

if __name__ == '__main__':
    from pdb import pm, set_trace
    # Sanity check for projections
    # V = np.array([[0, 0], [0, 1]])

    # cmplx = SimplicialComplex()
    # vertices = [cmplx.create_vertex(x) for x in V]
    # for i in range(len(V) - 1):
    #     cmplx.create_simplex([i, i+1])

    # obs_pts = np.array([[1, x] for x in np.linspace(-.3, 1.3)])
    # # set_trace()
    # # print cmplx.proj([1, -.3])

    # m_gt = BayesMesh1D(obs_pts=obs_pts, cmplx=cmplx)
    # m_gt.draw(block=True)
    
    V = np.array([[0, 0], [0, 1], [1, 2], [2, 1], [2, 0]])

    cmplx = SimplicialComplex()
    vertices = [cmplx.create_vertex(x) for x in V]
    for i in range(len(V) - 1):
        cmplx.create_simplex([i, i+1])

    
    m_gt = BayesMesh1D(cmplx=cmplx)
    # m_gt.draw(block=True)

    observed_pts = m_gt.sample_obs(500)

    m = BayesMesh1D(obs_pts = observed_pts)
    m.draw(block=True)
    
    
    # observed_pts = m_gt.sample_obs(500)
    # gt_ll = m_gt.log_likelihood()
    
    # # m_gt.draw(block=True)
    # # m_gt.mh(draw=50, samples=500)
    # # m_gt.draw(block=True)

    # V_init = V + mvn.rvs(np.zeros(2), 0.3 * np.eye(2), size=5)
    # print "V_init", V_init
    # m = Mesh1D(N=m_gt.N)
    # m.set_vertices(V_init)

    # m.set_obs(observed_pts, draw=False)
    # # m.draw(block=True)
    
    # # print m.log_likelihood()

    # m.mh(draw=10, samples=2000, gt_ll=gt_ll)
    

    
