from __future__ import division

import numpy as np

from bayesmesh import BayesMesh1D as bm

import h5py

################################################################################
##   Load Data
################################################################################
## replace with path to file of rope date
H5_FNAME = '/home/dhm/src/BayesGeom/data/overhand_actions.h5'
KEY_TEMPLATE = 'demo{}-seg00'
SUBKEY = 'cloud_xyz'

def get_clouds(fname, keys=None, N=1, proj_xy=True):
    f = h5py.File(fname, 'r')
    if keys is None:
        keys = [i+1 for i in range(N)]

    keys = [KEY_TEMPLATE.format(i) for i in keys]

    clouds = []

    for k in keys:
        cloud_k = f[k][SUBKEY][:]
        if proj_xy:
            cloud_k = cloud_k[:, :2]
        clouds.append(cloud_k)

    f.close()
    return clouds

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--init_only', action='store_true')
    parser.add_argument('--n_pts', type=int, default=100)
    parser.add_argument('--n_tests', type=int, default=1)
    parser.add_argument('--mcmc_steps', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    from pdb import pm, set_trace
    args=parse_arguments()
    observed_pts = get_clouds(H5_FNAME, N=args.n_tests, proj_xy=False)
    for obs_pts in observed_pts:
        if obs_pts.shape[0] > args.n_pts:
            indices = np.random.choice(range(obs_pts.shape[0]), 
                                       size=args.n_pts, replace=False)        
            obs_pts = obs_pts[indices, :]
        m = bm(obs_pts=obs_pts, n_clusters_init=args.n_clusters, d=3)
        if args.init_only:
            m.draw(block=True)
        else:    
            m.mh(draw=20, samples=args.mcmc_steps, final_block=False)
    
        
    
