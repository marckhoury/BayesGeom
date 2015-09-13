from __future__ import division

import numpy as np
from scipy.stats import expon, geom, multivariate_normal as mvn, dirichlet, norm
import scipy.spatial.distance as ssd
import scipy

from cvxpy import Variable, Problem, Parameter, sum_squares, Minimize
import matplotlib.pyplot as plt

from complex import Vertex, Simplex, SimplicialComplex
from registration import tps_cpd

from sklearn.cluster import KMeans

import time

eps = 1e-8

MAX_CACHE_SIZE=20

RJ_COR_ALPHA=100
COR_ALPHA=50
OBS_SIGMA=.1

P_RESTRICT_AFFINE = 0.3

## TPS kernel needs to be defined over a ball
## to be PSD
MAX_R = 100 ## should be big enough this doesn't matter

def tps_kernel1(distmat, d):
    if d == 1:
        return 1/12 * (2 * np.power(distmat, 3) -
                       3*MAX_R*np.power(distmat, 2) +
                       np.power(MAX_R, 3))
    elif d == 2:
        return (2 * np.power(distmat, 2)*np.log(distmat) -
                (1 + 2*np.log(MAX_R))*np.power(distmat, 2) + 
                np.power(MAX_R, 2))
    elif d == 3:
        ## looks like this is wrong?
        return (2 * np.power(distmat, 3) + 
                3 * MAX_R * np.power(distmat, 2) + 
                np.power(R, 3))
    else:
        raise NotImplemented, "tps kernel only defined for dimensions 1-3"
    

def tps_kernel(pt1, pt2):
    """
    Functional forms for dims 1-3 taken from 
    @article{williams2007gaussian,
             title={Gaussian process implicit surfaces},
             author={Williams, Oliver and Fitzgibbon, Andrew},
             journal={Gaussian Proc. in Practice},
             year={2007}
             }
    """
    d = pt1.shape[0]
    r = np.linalg.norm(pt1 - pt2) + 1e-16
    ## comment in to try RBF
    # return np.exp(-np.power(r, 2)/10)
    if d == 1:
        return 1/12 * (2 * np.power(r, 3) -
                       3*MAX_R*np.power(r, 2) +
                       np.power(MAX_R, 3))
    elif d == 2:
        return (2 * np.power(r, 2)*np.log(r) -
                (1 + 2*np.log(MAX_R))*np.power(r, 2) + 
                np.power(MAX_R, 2))
    elif d == 3:
        ## looks like this is wrong?
        return (2 * np.power(r, 3) + 
                3 * MAX_R * np.power(r, 2) + 
                np.power(R, 3))
    else:
        raise NotImplemented, "tps kernel only defined for dimensions 1-3"



class Obs(object):
    
    """
    latent variables for latent variable GP model

    eventually, these will generate the actual observations
    """
    
    def __init__(self, obs_pt, cmplx, latent_pt=None, sigma=OBS_SIGMA, s_source=None, proj=False):
        """
        pt: position of the latent point
        s: pointer to simplex that generated point
        """
        self.obs_pt = obs_pt

        self.d = obs_pt.shape[0]

        self.cmplx = cmplx
        # Simplex this observation is from
        ## sanity check

        if latent_pt is None:
            ## identity if no idea
            latent_pt = obs_pt
        self.s = s_source
        if self.s is None:
            _, _, self.s = cmplx.proj(latent_pt)
        
        # else:
        #     _, q = self.s.proj(latent_pt)

        assert self.s is not None
        if proj:
            latent_pt = self.s.proj(latent_pt)
        self.latent_pt = latent_pt
        # self.lc = self.s.local_coords(q)

    def __repr__(self):
        return "Obs({},{},{})".format(self.obs_pt, self.latent_pt, self.s)
            
    def set_source(self, s, lc=None):
        self.s = s
        # self.lc = lc

    # def latent_pt(self):
    #     return self.s.global_coords(self.lc)

    def draw(self, ax):
        gc = self.latent_pt
        ax.scatter(gc[0], gc[1])
        

class BayesMesh1D(object):

    """
    wrapper around a set of observations and a simplicial complex

    places a generic prior on complexes
    """
    def __init__(self, obs_pts=None, cmplx=None, 
                 gamma=.9, lmbda=.2, use_gp=True,
                 obs_sigma=OBS_SIGMA, propose_sigma=.0005, birth_sigma=.1,
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
        self.obs_sigma=obs_sigma
        self.obs_dist = norm(loc=0, scale=obs_sigma)

        self.birth_proposal = norm(loc=0, scale=birth_sigma)

        self.use_gp = use_gp

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
        chosen_simplices = []
        for i in range(n_samples):
            s = np.random.choice(simplices, p=p_simplex)
            chosen_simplices.append(s)
            lmbda = np.random.rand()
            pt_src = lmbda * s.vertices[0].v + (1-lmbda) * s.vertices[1].v
            # ## compute normal direction
            normal = s.vertices[0].v - s.vertices[1].v
            normal[0], normal[1] = normal[1], normal[0]
            normal = normal / np.linalg.norm(normal)
            normal[1] *= -1
            delta = self.obs_dist.rvs()
            pt = delta*normal + pt_src
            # print pt_src, pt, delta, normal
            pts[i, :] = pt

        C = np.eye(n_samples) * self.obs_sigma
        for i in range(n_samples):
            for j in range(n_samples):
                C[i, j] += tps_kernel(pts[i], pts[j])
        pts_obs = mvn(np.zeros(n_samples), C).rvs(size=2).T

        for i in range(n_samples):
            self.observations.append(Obs(pts_obs[i], self.cmplx, pts[i], s_source=chosen_simplices[i]))
        return pts, pts_obs, chosen_simplices

    def _set_obs(self, pts, latent_pts):
        self.observations = []
        n_obs = pts.shape[0]
        _, latent_pts = self.cmplx.proj_pts(latent_pts)
        for i in range(n_obs):
            o_i = Obs(pts[i], self.cmplx, latent_pt = latent_pts[i])
            self.observations.append(o_i)
        self.update_kernel_matrix()    

    #@profile
    def set_obs(self, pts, draw=False, n_resets=20):

        d = pts.shape[1]
        cmplx_pts = self.discretize_cmplx(pts_per_simplex=10)

        pts_h = np.c_[pts, np.ones(pts.shape[0])]

        kmeans = KMeans(init="k-means++", n_clusters=self.N)
        kmeans.fit_predict(pts)
        centroids = np.c_[kmeans.cluster_centers_, np.ones(self.N)]

        s_centers = []
        for s in self.cmplx.simplices.values():
            s_centers.append(s.global_coords([0.5]))
        s_centers = np.c_[np.array(s_centers), np.ones(self.N)]

        best_cost = np.inf
        best_R = None

        import time

        obs_sigma = self.obs_sigma
        self.obs_sigma = 1
    
        for n in range(n_resets):
            start = time.time()
            s_indices = range(self.N)
            k_indices = range(self.N)
            np.random.shuffle(s_indices)
            np.random.shuffle(k_indices)
            X = centroids[k_indices[:d+1]]
            Y = s_centers[s_indices[:d+1]]
            R = np.linalg.lstsq(X, Y)[0]
            pts_w = np.dot(pts_h, R)[:, :-1]
            self._set_obs(pts, pts_w)
            if draw:
                ## for debugging
                self.draw(block=True, show=False, outf='../figs/debug/registered_{}.png'.format(n))
            warped_cmplx = self.warp_cmplx()
            distmat = ssd.cdist(warped_cmplx, pts, 'sqeuclidean')
            cost = np.sum(np.min(distmat, axis=1))
            # cost = self.gp_ll()
            if cost < best_cost:
                best_cost = cost
                best_R = R

            print n, best_cost, cost, time.time() - start, -self.gp_ll()
        
        pts_w = np.dot(pts_h, best_R)[:, :-1]
        self.obs_sigma = obs_sigma
        self._set_obs(pts, pts_w)
        
        if draw:
            self.draw(block=True)

################################################################################
##     GP functions
################################################################################

    #@profile
    def update_kernel_matrix(self):
        self.X = np.zeros((len(self.observations), self.d))
        self.Y = np.zeros((len(self.observations), self.d))
        for i, o_i in enumerate(self.observations):
            self.X[i] = o_i.latent_pt
            self.Y[i] = o_i.obs_pt 
        if self.use_gp:
            self.C = self.obs_sigma * np.eye(len(self.observations)) + self.eval_kernel(self.X)
            self.inv_C = np.linalg.inv(self.C)

    #@profile
    def eval_kernel(self, U, X=None):
        if X is None:
            X = self.X
        distmat = ssd.cdist(U, X, 'euclidean') + 1e-16
        d = X.shape[1]
        res = tps_kernel1(distmat, d)        
        # import pdb; pdb.set_trace()
        return res
        # res = np.zeros((U.shape[0], X.shape[0]))
        # for i in range(U.shape[0]):
        #     for j in range(X.shape[0]):
        #         res[i, j] = tps_kernel(U[i, :], X[j, :])
        # return res

    def warp_pts(self, U):        
        C_ux = self.eval_kernel(U)        
        mu = C_ux.dot(self.inv_C).dot(self.Y)
        return mu

    def discretize_cmplx(self, pts_per_simplex=10):
        lmbdas = np.linspace(0, 1, 10)
        pts = []
        for s in self.cmplx.simplices.itervalues():
            v0 = s.vertices[0].v
            v1 = s.vertices[1].v
            for l in lmbdas:
                pts.append(l * v0 + (1-l) * v1)
        return np.array(pts)  

    def warp_cmplx(self, pts_per_simplex=10):
        pts = self.discretize_cmplx(pts_per_simplex)
        warped_pts = self.warp_pts(pts)
        return warped_pts

    def latent_obs_ll(self):
        ## don't include explicit simplex clusters
        _, pi = self.cmplx.proj_pts(self.X)
        ll = - np.power(self.X - pi, 2).sum() / (2*self.obs_sigma)
        return ll

    def guassian_ll(self):
        diff = self.Y - self.X
        return np.sqrt(np.trace(diff.T.dot(diff))) / self.obs_sigma


    #@profile
    def gp_ll(self):
        ll = np.trace(self.Y.T.dot(self.inv_C).dot(self.Y))
        return ll

    ##@profile
    def log_likelihood(self):
        self.update_kernel_matrix()
        prior_ll = self.prior_ll()
        latent_obs_ll = self.latent_obs_ll()
        if self.use_gp:
            obs_ll = self.gp_ll()
        else:
            obs_ll = self.gaussian_ll()
        # print 'prior_ll', prior_ll, 'obs_ll', obs_ll
        return prior_ll + latent_obs_ll + obs_ll

    # @profile
    def mh(self, samples=5000, draw=False, gt_ll=None, gt_structure_ll=None, final_block=False):
        if draw:
            fig, axarr = plt.subplots(2,2)
            axarr[0, 0].set_title('Observed')
            axarr[0, 1].set_title('Latent')
            axarr[1, 0].set_title('Log-Likelihood')
            axarr[1, 1].set_title('Stucture Log-Likelihood')
            log_likelihoods = [self.log_likelihood()]
            prior_ll = [self.prior_ll()]

            l_mcmc,  = axarr[1, 0].plot(range(len(log_likelihoods)), log_likelihoods, label='MCMC')

            if gt_ll:
                gt_ll_arr = np.ones(samples+1)*gt_ll
                l_gt,  = axarr[1, 0].plot(range(samples+1), gt_ll_arr, label='Ground Truth')
            axarr[1, 0].legend(loc='best')
            axarr[1, 0].set_xlim(0, samples+1)

            l_prior_ll,  = axarr[1, 1].plot(range(len(prior_ll)), prior_ll, label='Stucture_LL')
            if gt_structure_ll:
                gt_struct_ll_arr = np.ones(samples+1)*gt_structure_ll
                l_gt_struct = axarr[1, 1].plot(range(samples+1), gt_struct_ll_arr, label="GT_Structure_LL")

            axarr[1, 1].legend(loc='best')
            axarr[1, 1].set_xlim(0, samples+1)
            plt.show(block=False)
            plt.draw()
            # raw_input('go?')
                
        proposals = ['vertices', 'correspondence', 'death', 'birth', 'split', 'merge']
        accepts = {}
        for p in proposals:
            accepts[p] = (0, 0)
        proposal_p = [.3, .5, .05, .05, .05, .05]
        proposal_fns = {'vertices':self.propose_vertices, 
                     'correspondence':self.propose_correspondence, 
                     'death':self.propose_vertex_death, 
                     'birth':self.propose_vertex_birth,
                     'split':self.propose_vertex_split,
                     'merge':self.propose_vertex_merge}

        # proposal_p = [0, 0, 1, 0]
        accept = 0

        # print self.log_likelihood()
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
                if gt_structure_ll is not None:
                    axarr[1, 1].set_ylim(np.min(prior_ll), max(0, np.max(prior_ll), gt_structure_ll))
                else:
                    axarr[1, 1].set_ylim(np.min(prior_ll), max(0, np.max(prior_ll)))
                if gt_ll is not None:
                    axarr[1, 0].set_ylim(np.min(log_likelihoods), max(0, np.max(log_likelihoods)+50, gt_ll+50))
                else:
                    axarr[1, 0].set_ylim(np.min(log_likelihoods), max(0, np.max(log_likelihoods)+50))
                self.draw(block=False, latent_ax=axarr[0, 0], true_ax=axarr[0, 1])
                plt.draw()
                time.sleep(.1)
            for o in self.observations:
                assert o.s in self.cmplx.simplices.values()

        if draw:
            l_mcmc.set_data(range(len(log_likelihoods)), log_likelihoods)
            self.draw(block=final_block, latent_ax=axarr[0, 0], true_ax=axarr[0, 1])
            
        
    # @profile            
    def propose_vertices(self):
        ## similar to the way that we need to deal with the RJ steps
        ## symmetric
        ## pick random vertex
        v = np.random.choice(self.cmplx.vertices.values())
        v_old = v.v.copy()
        ## add random offset
        offset = self.propose_mvn.rvs()


        v_new = v_old + offset
        v_dist = np.linalg.norm(v_new - v_old)
        # print offset, v_new, v_dist
        if np.random.rand() > P_RESTRICT_AFFINE:
            v_star = tuple(self.cmplx.stars[v])
            s = np.random.choice(v_star)
            Q, o = s._affine_hull()
            q = v_new - o
            lc = np.dot(Q, q)
            v_new = s._pos_in_space(Q, lc) + o
            v_dist2 = np.linalg.norm(v_new - v_old)
            # print v_new, np.linalg.norm(v_new - v_old)
            if v_dist < v_dist2:
                print v_dist2 - v_dist
                import pdb; pdb.set_trace()


        def f_apply():           
            v.v = v_new
            ## proposal probabilities don't matter here
            return 0, 0

        def f_undo():
            v.v = v_old
            return 

        return (f_apply, f_undo)

    def propose_simplex(self, o):
        distances = self.cmplx.simplex_dists(o.latent_pt)
        
        simplices = []
        probs = np.zeros(len(distances))

        for i, s in enumerate(distances.keys()):
            simplices.append(s)
            probs[i] = self.obs_dist.pdf(distances[s])

        probs /= np.sum(probs)
        if np.any(np.isnan(probs)):
            ## fall back onto a uniform dist if 
            ## probs sum to 0
            probs = np.ones(len(distances)) * 1/ len(distances)

        s_new = np.random.choice(simplices, p=probs)
        res = dict(zip(simplices, probs))
        return res, s_new

    # @profile
    def propose_correspondence(self):
        ## project random point near an obs onto Mesh

        o = np.random.choice(self.observations)
        s_old = o.s

        probs, s_new = self.propose_simplex(o)
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
        len_ll = self.birth_proposal.logpdf(v.dist(kill_record['u']))

        ## probability we pick v's neighbor to birth v
        pick_v_neigh_ll = -np.log(len(self.cmplx.vertices) - 1)
        ## computes the steps of the birth_vertex method that
        ## invert the kill record and returns the log-likelihood
        birth_record = self.cmplx.birth_reverse(kill_record)
        birth_ll = self.cmplx.birth_ll(birth_record)

        ## compute observations to reassign to simplices
        obs_to_move = [o for o in self.observations if o.s in self.cmplx.stars[v]]
        # obs_to_move = self.observations
        obs_undo = []
        coresp_undo_ll = 0
        undo_lls = []
        for o in obs_to_move:
            obs_undo.append((o, o.s, o.s.get_key()))
            p_undo, _ = self.propose_simplex(o)
            coresp_undo_ll += np.log(p_undo[o.s])
            undo_lls.append(p_undo[o.s])

        def f_apply():
            old_ll = self.log_likelihood()
            self.cmplx.kill_vertex(kill_record=kill_record)
            coresp_apply_ll = 0
            apply_lls = []
            for o in obs_to_move:
                p_reassign, s_new = self.propose_simplex(o)
                coresp_apply_ll += np.log(p_reassign[s_new])
                apply_lls.append(p_reassign[s_new])
                o.s = s_new
            self.N -= 1            

            # new_ll = self.log_likelihood()
            # ll_forward = pick_v_ll + kill_ll + coresp_apply_ll
            # ll_backward = pick_v_neigh_ll + birth_ll + coresp_undo_ll + len_ll
            # ll_alpha = min(0, (new_ll + ll_backward) - (old_ll + ll_forward))    
            # print "apply_coresp:\t{}\tundo_coresp:\t{}".format(coresp_apply_ll, coresp_undo_ll)
            # print "death:\told_ll:\t{}\tnew_ll:\t{}\tforward:{}\tbackward:{}\taccept:\t{}".format(
            #     old_ll, new_ll, ll_forward, ll_backward, ll_alpha)

            # import pdb; pdb.set_trace()
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
        
        length = -1
        while length < 0:            
            length = self.birth_proposal.rvs()

        len_ll = self.birth_proposal.logpdf(length)
    
        birth_record = self.cmplx.birth_vertex(v, vec, length, persist=False)
        birth_ll = self.cmplx.birth_ll(birth_record)
        
        pick_v_new_ll = -np.log(len(self.cmplx.vertices) + 1)
        kill_record = self.cmplx.kill_reverse(birth_record)
        kill_ll = self.cmplx.kill_ll(kill_record)

        ## reassign observations that point to simplices 
        ## that will change
        obs_to_move = [o for o in self.observations if o.s in self.cmplx.stars[v]]
        # obs_to_move = self.observations
        # print v, len(obs_to_move)
        obs_undo = []
        coresp_undo_ll = 0
        for o in obs_to_move:
            obs_undo.append((o, o.s, o.s.get_key()))
            assert o.s in self.cmplx.simplices.values()
            p_undo, _ = self.propose_simplex(o)
            coresp_undo_ll += np.log(p_undo[o.s])
        
        def f_apply():
            # self.draw()
            # set_trace()
            old_ll = self.log_likelihood()
            self.cmplx.birth_vertex(birth_record=birth_record)
            coresp_apply_ll = 0
            for o in obs_to_move:
                p_reassign, s_new = self.propose_simplex(o)
                o.s = s_new
                coresp_apply_ll += np.log(p_reassign[s_new])
                # print p_reassign, s_new, self.cmplx.simplex_dists(o.pt)
            # self.log_likelihood()
            # self.draw()
            self.N += 1
            # new_ll = self.log_likelihood()
            # ll_forward = pick_v_ll + birth_ll + coresp_apply_ll + len_ll
            # ll_backward = pick_v_new_ll + kill_ll + coresp_undo_ll
            # ll_alpha = min(0, (new_ll + ll_backward) - (old_ll + ll_forward))    
            # print "birth:\told_ll:\t{}\tnew_ll:\t{}\tforward:{}\tbackward:{}\taccept:\t{}".format(
            #     old_ll, new_ll, ll_forward, ll_backward, ll_alpha)
            # import pdb; pdb.set_trace()


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
        holes = self.cmplx.merge_options()
        #hole = np.random.choice(options)
        ## ll of selecting this hole (uniformly from the holes)
        #hole_ll = -np.log(len(options))
        ## ll of selecting this particular u/v combo
        pick_v_u_ll = -np.log(len(holes)) - np.log(len(holes) - 1)
        v, u = np.random.choice(holes, 2, replace=False)

        merge_record = self.cmplx.merge_vertex(v, u, persist=False)
        merge_ll = self.cmplx.merge_ll(merge_record)

        ## probability we pick v's neighbor to birth v
        pick_v_new_ll = -np.log(len(self.cmplx.vertices) - 1)
        ## computes the steps of the birth_vertex method that
        ## invert the kill record and returns the log-likelihood
        split_record = self.cmplx.split_reverse(merge_record)
        split_ll = self.cmplx.split_ll(split_record)
        def f_apply():
            self.cmplx.merge_vertex(merge_record=merge_record)
            return hole_ll + pick_u_v_ll + merge_ll, pick_v_new_ll + split_ll

        def f_undo():
            self.cmplx.split_vertex(split_record=split_record)
            return

    def propose_vertex_split(self):
        ## reverse is vertex_mergeA
        hole = self.cmplx.merge_options()
        v = np.random.choice(self.cmplx.vertices.values())
        pick_v_ll = -np.log(len(self.cmplx.vertices) - len(holes))
        
        split_record = self.cmplx.split_vertex(v, persist=False) 
        split_ll = self.cmplx.split_ll(split_record)

        merge_record = self.cmplx.merge_reverse(split_record)
        merge_ll = self.cmplx.merge_ll(merge_record)

        def f_apply():
            ## need to compute probability of merging after changing the 
            ## complex
            self.cmplx.split_vertex(split_record=split_record)
            holes = self.cmplx.merge_options()
            pick_v_u_ll = -np.log(len(holes)) - np.log(len(holes)-1)
            #options = self.cmplx.merge_options()
            #hole_ll = -np.log(len(options))
            #for h in options:
            #    if v in h:
            #        pick_v_u_ll = -np.log(len(h)) - np.log(len(h)-1)
            #        break
            return pick_v_ll + split_ll, pick_v_u_ll + merge_ll

        def f_undo():
            self.cmplx.merge_vertex(merge_record=merge_record)

################################################################################
##     Drawing Utilities
################################################################################

    def warp_lines(self, low=-1, high=3):
        l_vals = np.linspace(low, high, 4)        
        y_vals = np.linspace(low, high, 30)
        lines = []
        warped_lines = []
        pts = np.zeros((30, 2))
        for l in l_vals:
            ## horizontal
            pts[:, 0] = l
            pts[:, 1] = y_vals
            lines.append(pts.copy())
            warped_lines.append(self.warp_pts(pts))

            ## vertical
            pts[:, 1] = l
            pts[:, 0] = y_vals
            lines.append(pts.copy())
            warped_lines.append(self.warp_pts(pts))
        return lines, warped_lines

        

    def draw(self, latent_ax=None, true_ax=None, show=True, block=False, outf=None):
        if latent_ax is None:
            fig, (latent_ax, true_ax) = plt.subplots(1, 2)

        latent_lines, true_lines = self.warp_lines()

        latent_ax.cla()
        latent_ax.set_title('Latent')
        for s in self.cmplx.simplices.itervalues():
            try:
                ((x0, y0), (x1, y1)) = s.vertices[0].v, s.vertices[1].v
            except ValueError:
                ((x0, y0, z0), (x1, y1, z0)) = s.vertices[0].v, s.vertices[1].v
            latent_ax.plot([x0, x1], [y0, y1], color='r')
        for l in latent_lines:
            latent_ax.plot(l[:, 0], l[:, 1], color='g')
        if self.observations:
            for o in self.observations:
                o.draw(latent_ax)        
        if true_ax is not None and self.observations:
            true_ax.cla()
            true_ax.set_title('Observed')
            for o in self.observations:
                true_ax.scatter(o.obs_pt[0], o.obs_pt[1], marker='o', color='b')
            warped_cmplx = self.warp_cmplx()
            true_ax.scatter(warped_cmplx[:, 0], warped_cmplx[:, 1], marker='x', color='r')
        for l in true_lines:
            true_ax.plot(l[:, 0], l[:, 1], color='g')
        if outf is not None:
            plt.savefig(outf, bbox_inches='tight')
        if show:
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
    m = create_house()
    obs = m.sample_obs
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
    # set_trace()
    # latent_pts, observed_pts, simplices = m_gt.sample_obs(100)

    # gt_ll = m_gt.log_likelihood()
    # gt_struct_ll = m_gt.prior_ll()
    # print gt_ll, gt_struct_ll
    # m_gt.draw(block=True, show=False, outf='../figs/debug/init.png')

    from static_ropetest import get_clouds, H5_FNAME
    obs_pts = get_clouds(H5_FNAME, keys=[7])[0]

    if obs_pts.shape[0] > 200:
        indices = np.random.choice(range(obs_pts.shape[0]), 
                                   size=200, replace=False)        
        obs_pts = obs_pts[indices, :]


    m_gt.set_obs(obs_pts)
    m_gt.draw(block=True, show=False, outf='../figs/debug/registered.png')
    print m_gt.log_likelihood()
    # m = BayesMesh1D(obs_pts = observed_pts, n_clusters_init=args.n_clusters)

    # m.mh(draw=20, gt_ll=gt_ll, gt_structure_ll=gt_struct_ll)


    
    
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
    

    
