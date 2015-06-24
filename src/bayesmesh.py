from __future__ import division

import numpy as np
from scipy.stats import expon, geom, multivariate_normal as mvn, dirichlet

from cvxpy import Variable, Problem, Parameter, sum_squares, Minimize

import matplotlib.pyplot as plt

import time

eps = 1e-8

class Simplex(object):

    def __init__(self, *vertices):
        """
        vertices: list of points
        """
        vertices = np.array(list(vertices), dtype=float)
        
        # k-dimensional simples
        self.k = len(vertices) - 1
        # D-dimensional space
        self.D = vertices[0].shape[0]
        assert self.k <= self.D, ("Cannot have a {}-dimensional simplex" +  
                                  "in a {}-dimensional space").format(self.k, self.D)
        self.vertices = vertices
        # pointers to neighbors
        # when filled in contains (s, i)
        # s is pointer to neighbor
        # i is index of self in s's neighbor list
        self.neighbors = [None for _ in self.vertices]

        self.lmbda_dist = dirichlet(np.ones(self.k+1))

        # setup QP to project into simplex
        # query point q
        # proj(q, self.vertices) = lmbda.dot(self.V)
        self.V = Parameter(self.D, self.k + 1)
        self.lmbda = Variable(self.k + 1)
        self.q = Parameter(self.D)
        f = lambda x: sum_squares(self.V * x - self.q)
        self.proj_opt = Problem(Minimize(f(self.lmbda)),
                                [eps <= self.lmbda, 1-eps >= self.lmbda,
                                 np.ones(self.k+1)*self.lmbda == 1])
        self.V.value = self.vertices

    def sample_lmbda(self, size=1):
        lmbda = self.lmbda_dist.rvs(size=size).T
        return lmbda

    def lmbda_ll(self, lmbda):
        return self.lmbda_dist.logpdf(lmbda)

    def global_coords(self, lmbda):
        return lmbda.T.dot(self.vertices).reshape((self.D, 1))
        
    # @profile
    def proj(self, query):
        self.q.value = query
        dist_sq = self.proj_opt.solve()
        lmbda = self.lmbda.value.reshape((self.D, 1))
        # ## clip values to deal with numerical issues
        # lmbda[lmbda > 1] = 1
        # lmbda[lmbda < 0] = 0
        q_proj = self.global_coords(lmbda)
        # print np.linalg.norm(q_proj - query), np.sqrt(dist_sq), q_proj, query
        # print self.vertices, self.V.value, lmbda
        # print 
        # print
        return q_proj, lmbda, np.linalg.norm(q_proj - query)

    def set_vertex(self, idx, val):
        self._set_vertex(idx, val, True)

    def _set_vertex(self, idx, val, set_neigh):
        self.vertices[idx, :] = val  
        assert np.all(self.vertices[idx, :] == val)
        if self.neighbors[idx] and set_neigh:
            n, idx_self = self.neighbors[idx]
            n._set_vertex(idx_self, val, False)
        self.V.value = self.vertices

    def set_neighbor(self, idx, neigh_info):
        if neigh_info is None:
            self.neighbors[idx] = None
            return
        (other, other_idx) = neigh_info            
        self.neighbors[idx] = (other, other_idx)
        other.neighbors[other_idx] = (self, idx)

    def check_neighbors(self):
        for i, n in enumerate(self.neighbors):
            if n:
                n_s, self_idx = n
                assert n_s.neighbors[self_idx] == (self, i)

    def __repr__(self):
        return "{}-dim Simplex({}, {})".format(self.k, self.vertices[0], self.vertices[1])

    def draw(self, ax):
        assert self.D <= 2
        for i in range(self.k+1):
            for j in range(self.k+1):
                if i == j: continue
                ((x_start, y_start), (x_end, y_end)) = (
                    self.vertices[i], self.vertices[j])
                X = [x_start, x_end]
                Y = [y_start, y_end]
                ax.plot(X, Y)

class ObsSrc(object):
    
    """
    latent variables for latent variable GP model

    eventually, these will generate the actual observations
    """
    
    def __init__(self, pt, sigma=.05, s_source=None, lmbda=None,
                 M = None):
        """
        pt: position of the point
        s_source: pointer to simplex that generated point
        lmbda: vector specifying coordinates on s_source
        """
        self.pt = pt
        self.M = M
        self.D = pt.shape[0]
        # Simplex this observation is from
        self.s = s_source
        # Local Coordinates on Simplex
        self.lmbda = lmbda
        self.dist = mvn(np.zeros(self.D), sigma*np.eye(self.D))
        if M is not None:
            self._init_s(M)
            
    # @profile
    def log_likelihood(self):
        ll = 0
        # import pdb; pdb.set_trace()
        pt_src = self.s.global_coords(self.lmbda)
        ll += self.dist.logpdf((self.pt - pt_src).T)
        assert not np.isnan(ll)
        ll += self.s.lmbda_ll(self.lmbda)
        assert not np.isnan(ll)        
        return ll

    def _init_s(self, M):
        self.s, self.lmbda = M.proj(self.pt)

    def resample_lmbda(self):
        self.lmbda = self.s.sample_lmbda()

    def draw(self, ax):
        ax.scatter(self.pt[0], self.pt[1], marker='o')
        pt_src = self.s.global_coords(self.lmbda)
        ((x_start, y_start), (x_end, y_end)) = (
            self.pt, pt_src)
        X = [x_start, x_end]
        Y = [y_start, y_end]
        ax.scatter(float(pt_src[0]), float(pt_src[1]), marker='v')
        ax.plot(X, Y, ls='--')
        
        

class Mesh1D(object):

    """
    Simplicial Complex
    """
    def __init__(self, gamma=.2, lmbda=.5, obs_sigma=.05, propose_sigma=.1,
                 D=2, N=None, P=None):
        """
        gamma: geometric variable for prior on number of simplices
        sigma_sq: variance of 
        d: dimension of embedding space
        """

        self.gamma = gamma
        self.N_prior = geom(gamma)
        if N is None:
            self.N = self.N_prior.rvs()
        else:
            self.N = N

        self.D = D
        self.lmbda = lmbda

        self.propose_mvn = mvn(np.zeros(self.D), propose_sigma*np.eye(self.D))

        self.obs_dist = mvn(np.zeros(self.D), obs_sigma*np.eye(self.D))

        p = np.zeros(D)
        self.simplices = []

        self.len_prior = expon(self.lmbda)

        s_lens = self.len_prior.rvs(size=self.N)

        ## directions are uniform dist over unit sphere
        s_dirs = mvn.rvs(np.zeros(self.D), np.eye(self.D), size=self.N)
        s_dirs = s_dirs.reshape((self.N, self.D))
        s_dirs /= np.sqrt(np.power(s_dirs, 2).sum(axis=1))[:, None]        
        s_old = None
        for i in range(self.N):
            p_next = p + s_dirs[i] * s_lens[i]
            s = Simplex(p, p_next)
            p = p_next            
            if s_old:
                s.set_neighbor(0, (s_old, 1))
            self.simplices.append(s)
            s_old = s

        self.obs = None    

    def print_vertices(self):
        s = self.simplices[0]
        while s.neighbors[0]:
            # find the start
            s, _ = s.neighbors[0]
        print_str = '( '
        while s.neighbors[1]:
            print_str += "{} ".format(s.vertices)
            s, _ = s.neighbors[1]
        print_str += "{} ".format(s.vertices)
        print_str += ")"
        print print_str

    def check_pointers(self):
        for s in self.simplices:
            s.check_neighbors()
               
    def prior_s_ll(self, s):
        s_len = np.linalg.norm(s.vertices[0] - s.vertices[1])
        # print 'simplex len', s_len
        ## add 1 b/c expon is a dist over [1, \infty)        
        ## and length can be [0, \infty)
        return self.len_prior.logpdf(s_len + 1)

    def prior_N_ll(self):
        return self.N_prior.logpmf(self.N)

    def prior_ll(self):
        # print 'simplex size prior', np.sum(self.prior_s_ll(s) for s in self.simplices)
        # print 'N prior ll', self.prior_N_ll()
        return (np.sum(self.prior_s_ll(s) for s in self.simplices) +
                self.prior_N_ll())

    def sample_obs(self, n_samples):
        self.obs = []
        pts = np.zeros((self.D, n_samples))
        for i in range(n_samples):
            s = np.random.choice(self.simplices)
            lmbda = s.sample_lmbda()
            pt_src = s.global_coords(lmbda)
            pt = self.obs_dist.rvs().reshape((self.D, 1)) + pt_src
            pts[:, i] = pt[:, 0]
            self.obs.append(ObsSrc(pt, s_source = s, lmbda=lmbda))
        return pts

    def set_obs(self, pts, draw=False):
        self.obs = []
        n_obs = pts.shape[1]
        for i in range(n_obs):
            pt = pts[:, i].reshape((self.D, 1))
            o_i = ObsSrc(pt, M=self)
            self.obs.append(o_i)
            if draw:
                self.draw(block=True)

    def obs_ll(self):
        ll = 0
        for o in self.obs:
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

            # for s in self.simplices:
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
        s = np.random.choice(self.simplices)
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

        o = np.random.choice(self.obs)
        lmbda_old = o.lmbda
        s_old = o.s

        offset = self.propose_mvn.rvs().reshape((self.D, 1))
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

        # print "proposing change to observation"
        # print "s_old", s_old, "s_new", s_new
        # ll_before = self.log_likelihood()
        # self.draw(block=False)
        # f_apply()
        # self.draw(block=False)
        # ll_after = self.log_likelihood()        
        # f_undo()
        # raw_input()
        # print "ll_before", ll_before, "ll_after", ll_after

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
    #     s = self.simplices[s_idx]

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


    """
    Geometric Operations
    """

    def proj(self, q):
        best_dist = np.inf
        best_s = None
        best_lmbda = None
        for s in self.simplices:
            s_q, s_lmbda, dist_sq = s.proj(q)
            # print s, q, dist_sq, s_q
            if np.sqrt(dist_sq) < best_dist:
                best_dist = np.sqrt(dist_sq)
                best_s = s
                best_lmbda = s_lmbda
        return best_s, best_lmbda

    """
    Topological Operations
    """

    def set_vertices(self, V):
        s = Simplex(V[0], V[1])
        self.simplices = [s]
        self.N = 1
        for i in range(1, V.shape[0]-1):
            s_new = Simplex(V[i], V[i+1])
            self.splice(s, s_new, 1)
            # self.print_vertices()
            s = s_new
            self.N += 1
        self.check_pointers()
        
    
    def splice(self, s, s_new, idx):
        if s.neighbors[idx]:
            s_new.set_neighbor(idx, s.neighbors[idx])
        s.set_neighbor(idx, (s_new, 1 - idx))
        self.simplices.append(s_new)
        self.N += 1

    def remove(self, s):
        if s.neighbors[0]:
            s_prv = s.neighbors[0][0]
            s_prv.set_neighbor(1, s.neighbors[1])
        elif s.neighbors[1]:
            s_nxt = s.neighbors[1][0]
            s_nxt.neighbors[0] = None
        self.simplices.remove(s)
        self.N -=1

    def draw(self, ax=None, block=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        for s in self.simplices:
            s.draw(ax)
        if self.obs:
            for o in self.obs:
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

def test_project():
    s1 = Simplex([0, 1], [1, 0])
    print s1.proj([2, 0])
    print s1.proj([0, 3])
    print s1.proj([1, 1])

if __name__ == '__main__':
    from pdb import pm, set_trace
    
    V = np.array([[0, 0], [0, 1], [1, 2], [2, 1], [2, 0]])
    # V = np.array([[0, 0], [0, 1], [1, 2]])
    
    
    m_gt = Mesh1D(N=1)
    m_gt.set_vertices(V)
    observed_pts = m_gt.sample_obs(500)
    gt_ll = m_gt.log_likelihood()
    
    # m_gt.draw(block=True)
    # m_gt.mh(draw=50, samples=500)
    # m_gt.draw(block=True)

    V_init = V + mvn.rvs(np.zeros(2), 0.3 * np.eye(2), size=5)
    print "V_init", V_init
    m = Mesh1D(N=m_gt.N)
    m.set_vertices(V_init)

    m.set_obs(observed_pts, draw=False)
    # m.draw(block=True)
    
    # print m.log_likelihood()

    m.mh(draw=10, samples=2000, gt_ll=gt_ll)
    

    
