from __future__ import division

import numpy as np
from scipy.stats import expon, geom, multivariate_normal as mvn, dirichlet, norm

from cvxpy import Variable, Problem, Parameter, sum_squares, Minimize
import matplotlib.pyplot as plt

from complex import Vertex, Simplex, SimplicialComplex

import time

eps = 1e-8

MAX_CACHE_SIZE=20

RJ_COR_ALPHA=50
COR_ALPHA=1

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
            _, _, self.s = cmplx.proj(self.pt)
        ## sanity check
        assert self.s is not None
        self._proj_cache = {}
        self.proj() 


    def proj(self):
        s_key = self.s.get_key()
        if s_key in self._proj_cache:
            self.dist, self.q = self._proj_cache[s_key]
        else:
            if len(self._proj_cache) > MAX_CACHE_SIZE:
                self._proj_cache.popitem()
            
            self.dist, self.q = self.s.proj(self.pt)
            self._proj_cache[s_key] = (self.dist, self.q)
        return self.dist, self.q

    def __repr__(self):
        return "Obs({},{})".format(self.pt, self.s)

    

            
    # @profile
    def log_likelihood(self):
        self.proj()
        ## prob of noise from source + probability of selecting source
        return self.noise_dist.logpdf(self.dist) - np.log(self.s.area())

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
    def __init__(self, obs_pts=None, cmplx=None, gamma=.99, lmbda=.5, obs_sigma=.05, propose_sigma=.005,
                 d=2, obs=None, N=None, P=None, n_clusters_init=5):
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
        self.obs_dist = norm(loc=0, scale=obs_sigma)

        self.cmplx = cmplx
        if self.cmplx is None:
            # obs_pts is not None
            self.cmplx = SimplicialComplex()
            ## this is a 1d complex
            self.cmplx.initialize(obs_pts, 1, n_clusters=n_clusters_init)

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
        return (np.sum(self.prior_s_ll(s) for s in self.cmplx.simplices.itervalues()) +
                self.prior_N_ll())

    def sample_obs(self, n_samples):
        self.observations = []
        pts = np.zeros((n_samples, self.d))
        simplices = self.cmplx.simplices.values()
        p_simplex = [s.area() for s in simplices]
        total_area = np.sum(p_simplex)
        p_simplex =[p/total_area for p in p_simplex]
        for i in range(n_samples):
            s = np.random.choice(simplices, p=p_simplex)
            lmbda = np.random.rand()
            pt_src = lmbda * s.vertices[0].v + (1-lmbda) * s.vertices[1].v
            ## compute normal direction
            normal = s.vertices[0].v - s.vertices[1].v
            normal[0], normal[1] = normal[1], normal[0]
            normal = normal / np.linalg.norm(normal)
            normal[1] *= -1
            delta = self.obs_dist.rvs()
            pt = delta*normal + pt_src
            # print pt_src, pt, delta, normal
            pts[i, :] = pt
            self.observations.append(Obs(pt, self.cmplx))
        return pts

    def set_obs(self, pts, draw=False):
        self.observations = []
        n_obs = pts.shape[0]
        for i in range(n_obs):
            pt = pts[i, :]
            o_i = Obs(pt, self.cmplx)
            self.observations.append(o_i)
        if draw:
            self.draw(block=True)

    def obs_ll(self):
        ll = 0
        simplices = self.cmplx.simplices.values()
        p_simplex = np.array([s.area() for s in simplices])
        p_simplex = np.log(p_simplex) - np.log(np.sum(p_simplex))
        p_simplex = dict(zip(simplices, p_simplex))
        for o in self.observations:
            ## need to compute distance to observation
            ll += o.log_likelihood()
            ll += p_simplex[o.s]
        return ll

    # @profile
    def log_likelihood(self):
        prior_ll = self.prior_ll()
        obs_ll = self.obs_ll()
        # print 'prior_ll', prior_ll, 'obs_ll', obs_ll
        return prior_ll + obs_ll

    # @profile
    def mh(self, samples=5000, draw=False, gt_ll=None, gt_structure_ll=None):
        if draw:
            fig, axarr = plt.subplots(3)
            axarr[1].set_title('Log-Likelihood')
            axarr[2].set_title('Stucture Log-Likelihood')
            log_likelihoods = [self.log_likelihood()]
            prior_ll = [self.prior_ll()]

            l_mcmc,  = axarr[1].plot(range(len(log_likelihoods)), log_likelihoods, label='MCMC')

            if gt_ll:
                gt_ll_arr = np.ones(samples+1)*gt_ll
                l_gt,  = axarr[1].plot(range(samples+1), gt_ll_arr, label='Ground Truth')
            axarr[1].legend(loc='best')

            l_prior_ll,  = axarr[2].plot(range(len(prior_ll)), prior_ll, label='Stucture_LL')
            if gt_structure_ll:
                gt_struct_ll_arr = np.ones(samples+1)*gt_structure_ll
                l_gt_struct = axarr[2].plot(range(samples+1), gt_struct_ll_arr, label="GT_Structure_LL")

            axarr[2].legend(loc='best')
            plt.show(block=False)
            plt.draw()
            # raw_input('go?')
                
        proposals = ['vertices', 'correspondence', 'death', 'birth']

        accepts = {}
        for p in proposals:
            accepts[p] = (0, 0)

        proposal_p = [.3, .6, .05, .05]

        proposal_fns = {'vertices':self.propose_vertices, 
                     'correspondence':self.propose_correspondence, 
                     'death':self.propose_vertex_death, 
                     'birth':self.propose_vertex_birth}

        # proposal_p = [0, 0, 1, 0]
        accept = 0

        print self.log_likelihood()
        for i in range(samples):
            propose_i = np.random.choice(proposals, p=proposal_p)

            # for s in self.cmplx.simplices:
            #     print s.vertices
            f_apply, f_undo = proposal_fns[propose_i]()
            # set_trace()
            accept_i = mh_step(self, f_apply, f_undo, verbose=False)   
            accepts[propose_i] = (accepts[propose_i][0] + accept_i, accepts[propose_i][1]+1)
            # print 'accepted', accept_i, self.log_likelihood()
            # if np.isnan(self.log_likelihood()):
            #     import pdb; pdb.set_trace()
            log_likelihoods.append(self.log_likelihood())
            prior_ll.append(self.prior_ll())
            
            # print self.log_likelihood(), propose_i, accept_i
            if draw and i%draw == 0:
                accept_str = ""
                for p, (accept_p, attempt_p) in accepts.iteritems():
                    if attempt_p == 0:
                        continue
                    accept_str += "{}:\t {:.3}\t".format(p, accept_p / attempt_p)
                print accept_str
                l_mcmc.set_data(range(len(log_likelihoods)), log_likelihoods)
                l_prior_ll.set_data(range(len(prior_ll)), prior_ll)
                axarr[2].set_ylim(np.min(prior_ll), max(0, np.max(prior_ll), gt_structure_ll))
                axarr[1].set_ylim(np.min(log_likelihoods), max(0, np.max(log_likelihoods)+50, gt_ll+50))
                self.draw(block=False, ax=axarr[0])
                plt.draw()
                time.sleep(.1)
            for o in self.observations:
                assert o.s in self.cmplx.simplices.values()

        if draw:
            l_mcmc.set_data(range(len(log_likelihoods)), log_likelihoods)
            self.draw(block=True, ax=axarr[0])
            
        
    # @profile            
    def propose_vertices(self):
        ## TODO: update to ensure no local overlaps
        ## similar to the way that we need to deal with the RJ steps
        ## symmetric
        ## pick random vertex
        v = np.random.choice(self.cmplx.vertices.values())
        v_old = v.v.copy()
        ## add random offset
        offset = self.propose_mvn.rvs()
        v_new = v_old + offset

        def f_apply():           
            v.v = v_new
            ## proposal probabilities don't matter here
            return 0, 0

        def f_undo():
            v.v = v_old
            return 

        return (f_apply, f_undo)

    def propose_simplex(self, o, alpha=10):
        distances = self.cmplx.simplex_dists(o.pt)
        
        simplices = []
        probs = np.zeros(len(distances))

        for i, s in enumerate(distances.keys()):
            simplices.append(s)
            probs[i] = np.exp(-alpha*distances[s])

        probs /= np.sum(probs)

        s_new = np.random.choice(simplices, p=probs)
        return dict(zip(simplices, probs)), s_new

    # @profile
    def propose_correspondence(self):
        ## project random point near an obs onto Mesh

        o = np.random.choice(self.observations)
        s_old = o.s

        probs, s_new = self.propose_simplex(o, alpha=COR_ALPHA)
        p_new = np.log(probs[s_new])
        p_old = np.log(probs[s_old])
    
        def f_apply():
            o.s = s_new
            return (p_new, p_old)

        def f_undo():
            o.s = s_old
            return

        return (f_apply, f_undo)



    ## Reversible Jump Proposals
    def propose_vertex_death(self):
        # import pdb; pdb.set_trace()
        if self.N == 1:
            return self.propose_vertices()
        ## reverse is vertex_birth
        v = np.random.choice(self.cmplx.vertices.values())
        ## probability of selecting v
        pick_v_ll = -np.log(len(self.cmplx.vertices))
        ## this just returns a record of the steps in the 
        ## kill move, and the log-likelihood of any arbitrary 
        ## decisions made
        kill_record = self.cmplx.kill_vertex(v, persist=False)
        kill_ll = self.cmplx.kill_ll(kill_record)

        ## likelihood of selecting that length
        len_ll = self.len_prior.logpdf(v.dist(kill_record['u']) + 1)

        ## probability we pick v's neighbor to birth v
        pick_v_neigh_ll = -np.log(len(self.cmplx.vertices) - 1)
        ## computes the steps of the birth_vertex method that
        ## invert the kill record and returns the log-likelihood
        birth_record = self.cmplx.birth_reverse(kill_record)
        birth_ll = self.cmplx.birth_ll(birth_record)

        ## compute observations to reassign to simplices
        # obs_to_move = [o for o in self.observations if o.s in self.cmplx.stars[v]]
        obs_to_move = self.observations
        obs_undo = []
        coresp_undo_ll = 0
        for o in obs_to_move:
            obs_undo.append((o, o.s, o.s.get_key()))
            p_undo, _ = self.propose_simplex(o, alpha=RJ_COR_ALPHA)
            coresp_undo_ll += np.log(p_undo[o.s])

        def f_apply():
            self.cmplx.kill_vertex(kill_record=kill_record)
            coresp_apply_ll = 0
            for o in obs_to_move:
                p_reassign, s_new = self.propose_simplex(o, alpha=RJ_COR_ALPHA)
                coresp_apply_ll += np.log(p_reassign[s_new])
                o.s = s_new
            self.N -= 1
            # set_trace()
            return (pick_v_ll + kill_ll + coresp_apply_ll, 
                    pick_v_neigh_ll + birth_ll + coresp_undo_ll + len_ll)

        def f_undo():
            self.cmplx.birth_vertex(birth_record=birth_record)
            replace_simps = {}
            for o, s_old, s_old_key in obs_undo:
                ## deal with the fact that pointers might point to dead
                ## simplices now                
                o.s = self.cmplx.get_simplex_by_key(s_old_key, replace_simps)
            self.N += 1
            return
        return (f_apply, f_undo)

    def propose_vertex_birth(self):
        # import pdb; pdb.set_trace()
        ## reverse is vertex_death
        v = np.random.choice(self.cmplx.vertices.values())
        pick_v_ll = -np.log(len(self.cmplx.vertices))
        
        vec = np.random.normal(size=(self.d,)) #generate a vector uniformily over the unit sphere
        length = np.linalg.norm(vec)
        vec /= length
        
        # subtract 1 b/c prior is over [1, \infty)
        length = self.len_prior.rvs() - 1
        len_ll = self.len_prior.logpdf(length + 1)
    
        birth_record = self.cmplx.birth_vertex(v, vec, length, persist=False)
        birth_ll = self.cmplx.birth_ll(birth_record)
        
        pick_v_new_ll = -np.log(len(self.cmplx.vertices) + 1)
        kill_record = self.cmplx.kill_reverse(birth_record)
        kill_ll = self.cmplx.kill_ll(kill_record)

        ## reassign observations that point to simplices 
        ## that will change
        # obs_to_move = [o for o in self.observations if o.s in self.cmplx.stars[v]]
        obs_to_move = self.observations
        # print v, len(obs_to_move)
        obs_undo = []
        coresp_undo_ll = 0
        for o in obs_to_move:
            obs_undo.append((o, o.s, o.s.get_key()))
            assert o.s in self.cmplx.simplices.values()
            p_undo, _ = self.propose_simplex(o, alpha=RJ_COR_ALPHA)
            coresp_undo_ll += np.log(p_undo[o.s])
        
        def f_apply():
            # self.draw()
            # set_trace()
            self.cmplx.birth_vertex(birth_record=birth_record)
            coresp_apply_ll = 0
            for o in obs_to_move:
                p_reassign, s_new = self.propose_simplex(o, alpha=RJ_COR_ALPHA)
                o.s = s_new
                coresp_apply_ll += np.log(p_reassign[s_new])
                # print p_reassign, s_new, self.cmplx.simplex_dists(o.pt)
            # self.log_likelihood()
            # self.draw(block=True)
            self.N += 1
            return (pick_v_ll + birth_ll + coresp_apply_ll + len_ll, 
                    pick_v_new_ll + kill_ll + coresp_undo_ll)

        def f_undo():
            self.cmplx.kill_vertex(kill_record=kill_record)
            replace_simps = {}
            for o, s_old, s_old_key in obs_undo:
                o.s = self.cmplx.get_simplex_by_key(s_old_key, replace_simps)
                assert o.s in self.cmplx.simplices.values()
            self.N -= 1
            return
        return (f_apply, f_undo)

    def propose_vertex_merge(self):
        ## reverse is vertex_split
        ## returns a list of holes
        ## where each hole is a set of vertices that could possibly 
        ## be merged
        options = self.cmplx.merge_options()
        hole = np.random.choice(options)
        ## ll of selecting this hole (uniformly from the holes)
        hole_ll = -np.log(len(options))
        ## ll of selecting this particular u/v combo
        pick_v_u_ll = -np.log(len(hole)) - np.log(len(hole) - 1)
        v, u = np.random.choice(hole, 2, replace=False)

        merge_record, merge_ll = self.cmplx.merge_vertex(v, u, persist=False)

        ## probability we pick v's neighbor to birth v
        pick_v_new_ll = -np.log(len(self.cmplx.vertices) - 1)
        ## computes the steps of the birth_vertex method that
        ## invert the kill record and returns the log-likelihood
        split_record, split_ll = self.cmplx.split_ll(merge_record=merge_record)
        def f_apply():
            self.cmplx.merge_vertex(merge_record=merge_record)
            return hole_ll + pick_u_v_ll + merge_ll, pick_v_new_ll + split_ll

        def f_undo():
            self.cmplx.split_vertex(split_record=split_record)
            return

    def propose_vertex_split(self):
        ## reverse is vertex_merge
        v = np.random.choice(self.cmplx.vertices.values())
        pick_v_ll = -np.log(len(self.cmplx.vertices))
        split_record, split_ll = self.cmplx.split_vertex(v, persist=False)

        merge_record, merge_ll = self.cmplx.merge_ll(split_record=split_record)

        def f_apply():
            ## need to compute probability of merging after changing the 
            ## complex
            self.cmplx.split_vertex(split_record=split_record)
            options = self.cmplx.merge_options()
            hole_ll = -np.log(len(options))
            for h in options:
                if v in h:
                    pick_v_u_ll = -np.log(len(h)) - np.log(len(h)-1)
                    break
            return pick_v_ll + split_ll, hole_ll + pick_v_u_ll + merge_ll

        def f_undo():
            self.cmplx.merge_vertex(merge_record=merge_record)

    def draw(self, ax=None, block=False):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        for s in self.cmplx.simplices.itervalues():
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

def create_house():
    V = np.array([[0, 0], [0, 1], [1, 2], [2, 1], [2, 0]])

    cmplx = SimplicialComplex()
    vertices = [cmplx.create_vertex(x) for x in V]
    for i in range(len(V) - 1):
        cmplx.create_simplex([i, i+1])
    
    m_gt = BayesMesh1D(cmplx=cmplx)
    return m_gt

def check_project():
    # Sanity check for projections
    V = np.array([[0, 0], [0, 1]])

    cmplx = SimplicialComplex()
    vertices = [cmplx.create_vertex(x) for x in V]
    for i in range(len(V) - 1):
        cmplx.create_simplex([i, i+1])

    obs_pts = np.array([[1, x] for x in np.linspace(-.3, 1.3)])
    # set_trace()
    # print cmplx.proj([1, -.3])

    m_gt = BayesMesh1D(obs_pts=obs_pts, cmplx=cmplx)
    m_gt.draw(block=True)
    

def check_init(obs_sizes=(20, 50, 100, 500)):
    m_gt = create_house()
    for n_samples in obs_sizes:
        observed_pts = m_gt.sample_obs(n_samples)
        m_gt.set_obs(observed_pts)
        m = BayesMesh1D(obs_pts = observed_pts)
        print "GT LL:\t{}".format(m_gt.log_likelihood())
        print "init LL:\t{}".format(m.log_likelihood())

        m_gt.draw(block=False)
        m.draw(block=True)

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_project', action='store_true')
    parser.add_argument('--check_init', action='store_true')
    parser.add_argument('--n_clusters', type=int, default=5)
    return parser.parse_args()

if __name__ == '__main__':
    from pdb import pm, set_trace
    args = parse_arguments()
    if args.check_project:
        check_project()

    if args.check_init:
        check_init()
    
    m_gt = create_house()
    observed_pts = m_gt.sample_obs(100)
    m_gt.set_obs(observed_pts)
    gt_ll = m_gt.log_likelihood()
    gt_struct_ll = m_gt.prior_ll()
    m = BayesMesh1D(obs_pts = observed_pts, n_clusters_init=args.n_clusters)

    m.mh(draw=20, gt_ll=gt_ll, gt_structure_ll=gt_struct_ll)


    
    
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
    

    
