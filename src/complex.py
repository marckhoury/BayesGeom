from __future__ import division

import numpy as np
import math

from scipy import linalg
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from predicates import orient2d

# number of digits to use for vertex keys
SIG_DIGITS=8

class Vertex(object):
    def __init__(self, coords):
        self.dim = len(coords)
        self.v = np.array(coords, dtype=np.float64).reshape((self.dim,))
        
    def dimension(self):
        return self.dim

    def index(self):
        return self.idx

    def set_index(self, i):
        self.idx = i
    
    def as_np_array(self):
        return self.v

    def dist(self, u):
        return np.linalg.norm(self.v - u.as_np_array())

    def translate(self, u):
        self.v = self.v + u

    def __getitem__(self, i):
        return self.v[i]

    def __iter__(self):
        return self.v.__iter__()

    def __repr__(self):
        return "V({}, {})".format(self.idx.__repr__(), self.get_key())
        #return tuple([x for x in self.v]).__repr__()

    def __eq__(self, other):
        return np.array_equal(self.v, other.v)
    
    def __hash__(self):
        return hash(self.idx)
    
    def get_key(self):
        return tuple(self.v.round(SIG_DIGITS))

class Simplex(object):
    def __init__(self, vertices):
        self.dim = len(vertices) - 1
        self.vertices = vertices
        
        self._orient_positively()
        
        self.indices = tuple([v.index() for v in self.vertices])
        self.index_set = frozenset(self.indices)
        self.neighbors = {}
    
    def dimension(self):
        return self.dim

    def index(self):
        return self.idx

    def set_index(self, i):
        self.idx = i

    def facet(self, i):
        if 0 <= i <= self.dim:
            return Simplex(self.vertices[:i] + self.vertices[i+1:])
        else:
            raise IndexError, '%s does not have facet %s.' % (self, i)
   
    def facets(self):
        return [self.facet(i) for i in range(self.dim + 1)]     

    def replace(self, v, u):
        self.vertices.remove(v)
        self.vertices.append(u)
    
        self._orient_positively()
        
        self.indices = tuple([w.index() for w in self.vertices])
        self.index_set = frozenset(self.indices)
            
        if v in self.neighbors:
            self.neighbors[u] = self.neighbors[v]
            del self.neighbors[v]

    def intersect(self, s):
        face_index_set = self.index_set.intersection(s.index_set)
        face_verts = [v for v in self.vertices if v.index() in face_index_set]
        return Simplex(face_verts)  
 
    def set_neighbor(self, v, s):
        self.neighbors[v] = s

    def set_neighbor_auto(self, s):
        f = self.index_set.intersection(s.index_set)
        if len(f) == self.dim:
            i = next(iter(self.index_set.difference(f)))
            i = self.indices.index(i)
            v = self.vertices[i]
            self.neighbors[v] = s
        if len(f) == s.dim:
            i = next(iter(s.index_set.difference(f)))
            i = s.indices.index(i)
            v = s[i]
            s.neighbors[v] = self

    def get_neighbor(self, v):
        try:
            return self.neighbors[v]
        except KeyError:
            return None
    
    def get_neighbors(self):
        return self.neighbors

    def local_coords(self, p):
        Q, o = self._affine_hull()
        q = p - o
        lc = np.dot(Q, q)
        return lc

    def global_coords(self, lc):
        Q, o = self._affine_hull()
        return self._pos_in_space(Q, lc) + o
    
    def proj(self, p):
        min_dist = np.inf
        min_point = None
        if self.dim == 0:
            v = self.vertices[0].as_np_array()
            min_dist = np.linalg.norm(p - v)
            min_point = v
        else:
            Q, o = self._affine_hull()
            q = p - o
            lc = np.dot(Q, q)
            ## seems like maybe this should be 
            ## self._pos_in_space + o? -- DHM
            gc = self._pos_in_space(Q, lc)
            d1 = np.linalg.norm(gc - q)
            if self._is_interior(Q, o, lc):
                if d1 < min_dist:
                    min_dist = d1
                    min_point = gc + o
            else:
                for f in self.facets():
                    d2, r = f.proj(gc + o)
                    d = math.sqrt(d1**2 + d2**2)
                    if d < min_dist:
                        min_dist = d
                        min_point = r
        return (min_dist, min_point)

    def proj_pts(self, P, min_dist=None, Q = None):
        if min_dist is None:
            min_dist = np.inf * np.ones(P.shape[0])
            Q = np.zeros(P.shape)
        
        if self.dim == 0:
            v = self.vertices[0].as_np_array()
            dist = np.linalg.norm(P - v, axis=1)
            idx = dist < min_dist
            min_dist[idx] = dist[idx]
            Q[idx, :] = v
            idx_change = idx
        else:
            lc, gc = self.proj_affine(P)
            D = np.linalg.norm(P - gc, axis=1)
            idx = D < min_dist
            interior = np.all(lc >= 0, axis=1)
            idx_change = idx * interior
            min_dist[idx_change] = D[idx_change]
            Q[idx_change, :] = gc[idx_change, :]
            cont_idx = idx * np.logical_not(interior)
            mind_cont = min_dist[cont_idx]
            Q_cont = Q[cont_idx, :]
            for f in self.facets():
                d_f, q_f, idx_chng_f = f.proj_pts(gc[cont_idx], min_dist[cont_idx], Q[cont_idx, :])
                D2 = np.sqrt(np.power(D[cont_idx], 2) + np.power(d_f, 2))
                idx = D2 < mind_cont
                mind_cont[idx] = D2[idx]
                Q_cont[idx, :] = q_f[idx]
                idx_change[cont_idx] = idx
            min_dist[cont_idx] = mind_cont
            Q[cont_idx, :] = Q_cont
        return min_dist, Q, idx_change

    def proj_affine(self, P):
        d = P.shape[1]
        N = P.shape[0]
        V = np.array([self.vertices[i].as_np_array() for i in range(self.dim+1)]).T
        Q_i = V.T.dot(V)
 
        Q = linalg.block_diag(*[Q_i for i in range(N)])
        
        q = - np.reshape(V.T.dot(P.T).T, (N * (self.dim + 1)))
        A_i = np.ones(self.dim + 1)
        A = linalg.block_diag(*[A_i for i in range(N)])

        ## Z * [alpha; lambda].T = c
        ## lhs of KKT
        n_vars = N * (self.dim + 1)
        n_cnts = N        
        Z = np.zeros((n_vars + n_cnts, n_vars  + n_cnts))
        Z[:n_vars, :n_vars] = Q
        Z[n_vars:, :n_vars] = A
        Z[:n_vars, n_vars:] = A.T

        ## rhs of KKT
        c = np.zeros(n_vars + n_cnts)
        c[:n_vars] = -q
        c[n_vars:] = np.ones(n_cnts)

        alpha = np.linalg.solve(Z, c)
        alpha = alpha[:n_vars].reshape(N, self.dim + 1)
        P_affine = alpha.dot(V.T)
        return alpha, P_affine
            
    def _is_interior(self, Q, o, lc):
        if self.dim == 1:
            v1 = self.vertices[0].as_np_array()
            v2 = self.vertices[1].as_np_array()
            v1, v2 = v1 - o, v2 - o
            v1, v2 = np.dot(Q, v1), np.dot(Q, v2)
            return v1 <= lc <= v2
        elif self.dim == 2:
            v1 = self.vertices[0].as_np_array()
            v2 = self.vertices[1].as_np_array()
            v3 = self.vertices[2].as_np_array()
            v1, v2, v3 = v1 - o, v2 - o, v3 - o
            v1, v2, v3 = np.dot(Q, v1), np.dot(Q, v2), np.dot(Q, v3)
            return orient2d(v1, v2, lc) >= 0 \
                    and orient2d(v2, v3, lc) >= 0 \
                    and orient2d(v3, v1, lc) >= 0
        elif self.dim == 3:
            v1 = self.vertices[0].as_np_array()
            v2 = self.vertices[1].as_np_array()
            v3 = self.vertices[2].as_np_array()
            v4 = sefl.vertices[3].as_np_array()
            v1, v2 = v1 - o, v2 - o
            v3, v3 = v3 - o, v4 - o
            v1, v2 = np.dot(Q, v1), np.dot(Q, v2)
            v3, v4 = np.dot(Q, v3), np.dot(Q, v4)
            return orient3d(v1, v2, v3, lc) >= 0 \
                    and orient3d(v1, v3, v4, lc) >= 0 \
                    and orient3d(v1, v4, v2, lc) >= 0 \
                    and orient3d(v2, v4, v3, lc) >= 0
        else:
            raise Exception, 'Operation _is_interior only supports simplices of dimension 1-3.'

    def _pos_in_space(self, Q, lc):
        gc = np.zeros((Q.shape[1],))
        for i in xrange(0, Q.shape[0]):
            gc += lc[i]*Q[i,:]
        return gc

    def _affine_hull(self):
        X = np.array([v.as_np_array() for v in self.vertices[1:]])
        X = np.transpose(X - self.vertices[0].as_np_array())
        Q = np.transpose(linalg.orth(X))
        return (Q, self.vertices[0].as_np_array())
        
    def _orient_positively(self):
        if self.dim == 2:
            v1 = self.vertices[0].as_np_array()
            v2 = self.vertices[1].as_np_array()
            v3 = self.vertices[2].as_np_array()
            if orient2d(v1, v2, v3) < 0:
                self.vertices[1], self.vertices[2] = self.vertices[2], self.vertices[1]
        elif self.dim == 3:
            v1 = self.vertices[0].as_np_array()
            v2 = self.vertices[1].as_np_array()
            v3 = self.vertices[2].as_np_array()
            v4 = self.vertices[3].as_np_array()
            if orient3d(v1, v2, v3, v4) < 0:
                self.vertices[2], self.vertices[3] = self.vertices[3], self.vertices[2]
        elif self.dim >= 4:
            A = np.array([v.as_np_array() for v in self.vertices])
            one_col = np.ones((len(self.vertices), 1))
            A = np.append(A, one_col, 1)
            if linalg.det(A) < 0:
                self.vertices[-2], self.vertices[-1] = self.vertices[-1], self.vertices[-2]
        
    def __contains__(self, s):
        if isinstance(s, Vertex):
            return s.index() in self.index_set
        else:
            return s.index_set.issubset(self.index_set)
    
    def __getitem__(self, i):
        return self.vertices[i]

    def __iter__(self):
        return self.vertices.__iter__()

    def __repr__(self):
        return 'Simplex{}'.format(self.indices)
    
    def __eq__(self, other):
        return self.index_set == other.index_set

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return self.index_set.__hash__()

    def get_key(self):
        return tuple(sorted(v.get_key() for v in self))

    def area(self):
        if self.dim == 1:
            return self.vertices[0].dist(self.vertices[1])
        raise NotImplemented

class SimplicialComplex(object):
    def __init__(self):
        self.simplices = {}
        self.simplex_set = set()
        self.vertices = {}
        ## Maps vertices to the set of simplices
        ## that contain it
        self.stars = {}
        self.holes = []
        self.next_vertex_index = 0
        self.next_simplex_index = 0
        self.dim = 0

    def create_vertex(self, coords):
        v = Vertex(coords)
        v.set_index(self.next_vertex_index)
        self.vertices[self.next_vertex_index] = v
        self.stars[v] = set()
        self.next_vertex_index += 1
        return v

    def create_simplex(self, vertex_indices):
        simp_verts = [self.vertices[i] for i in vertex_indices]
        s = Simplex(simp_verts)
        if s not in self.simplex_set:
            s.set_index(self.next_simplex_index)
            self.simplices[self.next_simplex_index] = s
            for v in simp_verts:
                self.stars[v].add(s)
            self.next_simplex_index += 1
            self.simplex_set.add(s)
            
            if self.dim < len(vertex_indices) - 1:
                self.dim = len(vertex_indices) - 1
            
            return s
        else:
            return None #Need to fix this, but doesn't seem like any call makes use of return value

    def get_simplex_by_key(self, s_key, replace_dict=None):
        # import pdb; pdb.set_trace()
        if replace_dict is not None and s_key in replace_dict:
            return replace_dict[s_key]
        for t in self.simplices.itervalues():
            if t.get_key() == s_key:
                replace_dict[s_key] = t
                return t
        raise KeyError
            
    
    def get_vertex(self, i):
        try:
            return self.vertices[i]
        except KeyError:
            return None

    def get_simplex(self, i):
        try:
            return self.simplices[i]
        except KeyError:
            return None
    
    def vertex_count(self):
        return len(self.vertices)
        
    def simplex_count(self):
        return len(self.simplices)
    
    def itervertices(self):
        return self.vertices.itervalues()

    def itersimplices(self):
        return self.simplices.itervalues()

    def dimension(self):
        return self.dim

    def proj(self, p):
        min_dist = np.inf
        min_point = None
        min_s = None
        for s in self.simplices.itervalues():
            d, q = s.proj(p)
            if d < min_dist:
                min_dist = d
                min_point = q
                min_s = s
        return (min_dist, min_point, min_s)

    def proj_pts(self, P):
        min_dist = np.inf * np.ones(P.shape[0])
        min_pts = np.zeros(P.shape)
        for s in self.simplices.itervalues():
            min_dist, min_pts, _ = s.proj_pts(P, min_dist = min_dist, Q = min_pts)
        return min_dist, min_pts
            

    def simplex_dists(self, p):
        res = {}
        for s in self.simplices.itervalues():
            res[s] = s.proj(p)[0]
        return res
    
    def star(self, s):
        return [t for t in self.simplices.itervalues() if s in t]
    
    def link(self, s):
        link_s = []
        for t in self.star(s):
            for f in t.facets():
                if s not in f:
                    link_s.append(f)
        return link_s

    def initialize(self, points, d,n_clusters=8):
        n = points.shape[0]
        kmeans = KMeans(init="k-means++", n_clusters=int(n_clusters))
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        for i in xrange(0, centroids.shape[0]):
            centroid = centroids[i]
            self.create_vertex(centroid)
        if d == 1:
            self._construct_curve(centroids)
        elif d == 2:
            self._construct_surface(centroids)
        elif d == 3:
            self._construct_volume(centroids)
        else:
            raise Exception, 'initialize procedure only implemented for simplicial complexes with dimension 1-3.'
        ## remove any leftover vertices that aren't in simplices
        to_del = []
        for i, v in self.vertices.iteritems():
            used=False
            for s in self.simplices.itervalues():
                if v in s:
                    used=True
                    break
            if not used:
                to_del.append(v)
        for v in to_del:
            del self.stars[v]
            del self.vertices[v.index()]

    def kill_vertex(self, v=None, persist=True, kill_record=None):
        if kill_record is not None and 'on_the_fly' in kill_record:
            v = self.vertices[self.next_vertex_index - 1]
            persist = True
            kill_record = None
        
        if kill_record is not None:
            self._apply_kill_record(kill_record)
        else:
            kill_record = {}
            kill_record['v'] = v

            if len(self.stars[v]) > 0:
                sv = next(iter(self.stars[v]))
                u = sv[0] if sv[0].index() != v.index() else sv[1]
                kill_record['u'] = u
                kill_record['star_size'] = len(self.stars[v]) * self.dim
                
                destroyed_simplices = self.stars[v].intersection(self.stars[u])
                kill_record['destroyed_simplices'] = destroyed_simplices
            
                kill_record['neighbor_updates'] = []
                for s in destroyed_simplices:
                    vn = s.get_neighbor(v)
                    un = s.get_neighbor(u)
                    w, y, = None, None 
                    if vn is not None:
                        w = next(iter(vn.index_set.difference(s.index_set)))
                        w = vn.indices.index(w)
                        w = vn[w]
                    if un is not None:
                        y = next(iter(un.index_set.difference(s.index_set)))
                        y = un.indices.index(y)
                        y = un[y]
                
                    kill_record['neighbor_updates'].append((vn, w, un, y)) 
            else:
                kill_record['u'] = None
                kill_record['star_size'] = 0
                kill_record['destroyed_simplices'] = []
                kill_record['neighbor_updates'] = []
            
            if persist:
                self._apply_kill_record(kill_record)
            return kill_record
    
    def kill_reverse(self, birth_record):
        #Constructing this kill_record is a little difficult
        #sice the vertex to kill hasn't been created yet, in this case
        #so I'm going to do something hacky
        #I'll create a kill_record with enough info to compute the log likelihood
        #based on the state assuming the birth_record has been applied
        #Then I'll include a field called 'on_the_fly' so when 
        #apply_kill_record is called with this record
        #it will compute the appropriate kill_record then and there
        #using the most recently created vertex
        kill_record = {}
        kill_record['on_the_fly'] = True
        
        #size of the star of the newly created vertex
        star_size = len(birth_record['new_simplices']) + len(birth_record['update_simplices']) 
    
        #number of vertices to possibly collapse into for the newly created star
        kill_record['star_size'] = star_size * self.dim
        
        return kill_record
    
    def kill_ll(self, kill_record):
        return -np.log(kill_record['star_size'])
    
    def birth_vertex(self, v=None, vec=None, length=None, persist=True, birth_record=None):
        if birth_record is not None:
            self._apply_birth_record(birth_record)
        else:
            birth_record = {}
            birth_record['v'] = v
            birth_record['dir'] = vec
            birth_record['star_size'] = len(self.stars[v])
            birth_record['new_simplices'] = []
            birth_record['update_simplices'] = []
            
            min_dist = np.inf    
            for s in self.stars[v]:
                for u in s:
                    dist = v.dist(u)
                    if u.index() != v.index() and dist < min_dist:
                        min_dist = dist
            birth_record['length'] = length if length <= min_dist else min_dist
 
            pos = v.as_np_array() + birth_record['length'] * vec
            
            min_dist = np.inf
            min_simp = None 
            for s in self.stars[v]:
                dist, _ = s.proj(pos)
                if dist < min_dist:
                    min_dist = dist
                    min_simp = s
           
            if min_simp is not None:
                u = min_simp[0] if min_simp[0].index() != v.index() else min_simp[1]
                to_update = self.stars[v].intersection(self.stars[u])
                for s in to_update:
                    birth_record['update_simplices'].append(s)
                    index_set = [w.index() for w in s if w.index() != u.index()]
                    birth_record['new_simplices'].append(index_set)
            else:
                birth_record['new_simplices'].append([v.index()])     
         
            if persist:
                self._apply_birth_record(birth_record)
            return birth_record

    def birth_reverse(self, kill_record):
        birth_record = {}
        v, u = kill_record['v'], kill_record['u']
        birth_record['v'] = u
        birth_record['star_size'] = len(self.stars[v])
         
        vec = v.as_np_array() - u.as_np_array()
        length = np.linalg.norm(vec)
        birth_record['dir'] = vec/length
        birth_record['length'] = length
      
        birth_record['new_simplices'] = []
        for s in kill_record['destroyed_simplices']:
            index_set = [w.index() for w in s if w.index() != v.index()]
            birth_record['new_simplices'].append(index_set) 
    
        birth_record['update_simplices'] = [] 
        for s in kill_record['destroyed_simplices']:
            t = s.get_neighbor(kill_record['u'])
            if t is not None:
                birth_record['update_simplices'].append(t)
                
        return birth_record
 
    def birth_ll(self, birth_record):
        return -np.log(birth_record['star_size']) 
    
    def merge_options(self):
        if self.dim == 1:
            return self.holes[0]
    
    def merge_vertex(self, v=None, u=None, persist=False, merge_record=None):
        if merge_record is not None:
            self._apply_merge_record(merge_record)
        elif self.dim == 1:
            merge_record = {}
            merge_record['v'] = v
            merge_record['u'] = u
            
            sv = self.stars[v][0]
            su = self.stars[u][0]
            
            nv = sv[0] if sv[0].index() != v.index() else sv[1]
            nu = su[0] if su[0].indeX() != u.index() else nu[1]
            
            merge_record['neighbor_updates'].append((sv, nv, su, nu))
            
            if persist:
                self._apply_merge_record(merge_record)
            return merge_record
    
    def merge_ll(self, merge_record):
        pass
    
    def split_vertex(self, v, persist=False, split_record=None):
        if split_record is not None:
            self._apply_split_record(split_record)
        elif self.dim == 1:
            split_record = {}
            split_record['v'] = v
            split_record['s'] = self.stars[v][0]
            split_record['t'] = self.stars[v][1] if len(self.stars[v]) > 1 else None
            split_record['star_size'] = len(self.stars[v]) 
             
            if persist:
                self._appply_split_record(split_record)
            return split_record
    
    def split_ll(self, split_record):
        pass  
        
    def translate_vertex(self, v, coords):
        v.translate(coords)
  
    def _apply_kill_record(self, kill_record):
        for update in kill_record['neighbor_updates']:
            vn, w, un, x = update
            if vn is not None:
                vn.set_neighbor(w, un)
            if un is not None:
                un.set_neighbor(x, vn)

        for s in kill_record['destroyed_simplices']:
            for w in s:
                self.stars[w].remove(s)
            t = s.get_neighbor(kill_record['u'])
            if t is not None:
                for x in t:
                    self.stars[x].remove(t)
                t.replace(kill_record['v'], kill_record['u'])
                for x in t:
                    self.stars[x].add(t)
            del self.simplices[s.index()]
        del self.stars[kill_record['v']]    
        del self.vertices[kill_record['v'].index()]
            
    def _apply_birth_record(self, birth_record):
        v = birth_record['v']
        u = v.as_np_array() + birth_record['length'] * birth_record['dir']
        u = self.create_vertex(u) 
        
        for s in birth_record['new_simplices']:
            s.append(u.index())
            self.create_simplex(s)
    
        for s in birth_record['update_simplices']:
            for w in s:
                self.stars[w].remove(s)
            s.replace(v, u)
            for w in s:
                self.stars[w].add(s)
    
        modified = self.stars[v].union(self.stars[u])
        
        for s in modified:
            for t in modified:
                s.set_neighbor_auto(t)
    
    def _apply_merge_record(self, merge_record):
        for update in merge_record['neighbor_updates']:
            sv, nv, su, nu = update
            sv.set_neighbor(nv, su)
            su.set_neighbor(nu, sv)
            su.replace(nu, nv)
            del self.vertices[nu.index()]
            del self.stars[nu]

    def _apply_split_record(self, split_record):
        v = split_record['v']
        v_coords = v.as_np_array()
        w = self.create_vertex(v_coords)
        for update in split_record['neighbor_updates']:
            sv, nv, su, nu = update
            sv.set_neighbor(nv, None)
            su.set_neighbor(nu, None)
            su.replace(v, w)
            self.stars[w].add(su)
            self.stars[v].remove(su)
    
    def _construct_curve(self, points):
        delaunay = Delaunay(points)
        indices, indptr = delaunay.vertex_neighbor_vertices
        edge_table = []
        for i in xrange(0, points.shape[0]):
            neighbors = indptr[indices[i]:indices[i+1]]
            edges = self._get_curve_edges(points, i, neighbors)
            edge_table.append(edges)
        for i in xrange(0, points.shape[0]):
            for j in edge_table[i]:
                if j != None and i in edge_table[j]:
                    self.create_simplex([i,j])
        for s in self.simplices.values():
            for t in self.simplices.values():
                s.set_neighbor_auto(t)
             
            
    def _get_curve_edges(self, points, i, neighbors):
        min_dist = np.inf
        min_neighbor = None
        p = points[i]
        for neighbor in neighbors:
            neighbor_pos = points[neighbor]
            dist = np.linalg.norm(p - neighbor_pos)
            if dist < min_dist:
                min_dist = dist
                min_neighbor = neighbor
        vec1 = points[min_neighbor] - p
        min_dist = np.inf
        min_half_neighbor = None
        for neighbor in neighbors:
            neighbor_pos = points[neighbor]
            dist = np.linalg.norm(p - neighbor_pos)
            vec2 = neighbor_pos - p
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = math.acos(cos_theta)
            # print p, neighbor_pos, theta
            if theta > math.pi / 2 and dist < min_dist:
                min_dist = dist
                min_half_neighbor = neighbor
        
        return (min_neighbor, min_half_neighbor)
    
    def _construct_surface(self, points):
        pass

    def _construct_volume(self, points):
        pass 
    
if __name__ == '__main__':
    v1 = Vertex([0,0,0])
    v2 = Vertex([1,0,0])
    v3 = Vertex([0.5,1,0])
    v1.set_index(1)
    v2.set_index(2)
    v3.set_index(3)
    print v1, v1.index(), v1.as_np_array(), v1.dist(v2)
    s = Simplex([v1,v3,v2])
    t = Simplex([v1,v2,v3])
    print s.intersect(t)
    s.set_neighbor_auto(t)
    print s.get_neighbors()
    print t.get_neighbors()
    v = Vertex([0,1,3,1])
    print s.facet(0)
    print v[1]
    p = np.array([0.5,0.5,100])
    print s.proj(p)
    p = v1.as_np_array()
    print s.proj(p)
    p = v2.as_np_array()
    print s.proj(p)
    p = v3.as_np_array()
    print s.proj(p)
    p = np.array([1,1,100])
    print s.proj(p)
    p = np.array([1,1,0])
    print s.proj(p)
    p = np.array([0.5, 0, 10])
    print s.proj(p)
    cmplx = SimplicialComplex()
    v4 = cmplx.create_vertex(v1.as_np_array())
    cmplx.create_vertex(v2.as_np_array())
    cmplx.create_vertex(v3.as_np_array())
    s2 = cmplx.create_simplex([0,2,1])
    print s2 
    print cmplx.proj(p)
    print cmplx.star(v4)
    print cmplx.link(v4)
    print cmplx.star(cmplx.link(v4)[0])
