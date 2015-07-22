from __future__ import division

import numpy as np
import math

from scipy import linalg
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans
from predicates import orient2d

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

    def __getitem__(self, i):
        return self.v[i]

    def __iter__(self):
        return self.v.__iter__()

    def __repr__(self):
        return tuple([x for x in self.v]).__repr__()

class Simplex(object):
    def __init__(self, vertices):
        self.dim = len(vertices) - 1
        self.vertices = vertices 
        
        self._orient_positively()
        
        self.indices = tuple([v.index() for v in self.vertices])
        self.index_set = frozenset(self.indices)
        self.neighbors = [None] * (self.dim + 1)
    
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

    def intersect(self, s):
        face_index_set = self.index_set.intersection(s.index_set)
        face_verts = [v for v in self.vertices if v.index() in face_index_set]
        return Simplex(face_verts)  
 
    def set_neighbor(self, s, i):
        self.neighbors[i] = s

    def set_neighbor(self, s):
        f = self.index_set.intersection(s.index_set)
        if len(f) == self.dim:
            i = next(iter(self.index_set.difference(f)))
            i = self.vertices.index(i)
            self.neighbors[i] = s
        if len(f) == s.dim:
            i = next(iter(s.index_set.difference(f)))
            i = s.vertices.index(i)
            s.neighbors[i] = self

    def get_neighbor(self, i):
        return self.neighbors[i]
    
    def get_neighbors(self):
        return list(self.neighbors)
    
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

class SimplicialComplex(object):
    def __init__(self):
        self.simplices = []
        self.vertices = []
        self.next_vertex_index = 0
        self.next_simplex_index = 0
        self.dim = 0

    def create_vertex(self, coords):
        v = Vertex(coords)
        v.set_index(self.next_vertex_index)
        self.next_vertex_index += 1
        self.vertices.append(v)
        return v

    def create_simplex(self, vertex_indices):
        simp_verts = [self.vertices[i] for i in vertex_indices]
        s = Simplex(simp_verts)
        s.set_index(self.next_simplex_index)
        self.next_simplex_index += 1
        self.simplices.append(s)
    
        if self.dim < len(vertex_indices) - 1:
            self.dim = len(vertex_indices) - 1
        
        return s
    
    def get_vertex(self, i):
        return self.vertices[i]

    def get_simplex(self, i):
        return self.simplices[i]
    
    def vertex_count(self):
        return len(self.vertices)
        
    def simplex_count(self):
        return len(self.simplices)
    
    def itervertices(self):
        return self.vertices.__iter__()

    def itersimplices(self):
        return self.simplices.__iter__()

    def dimension(self):
        return self.dim

    def valid(self):
        pass
    
    def proj(self, p):
        min_dist = np.inf
        min_point = None
        min_s = None
        for s in self.simplices:
            d, q = s.proj(p)
            if d < min_dist:
                min_dist = d
                min_point = q
                min_s = s
        return (min_dist, min_point, min_s)
    
    def star(self, s):
        return [t for t in self.simplices if s in t]
    
    def link(self, s):
        link_s = []
        for t in self.star(s):
            for f in t.facets():
                if s not in f:
                    link_s.append(f)
        return link_s

    def initialize(self, points, d):
        n = points.shape[0]
        kmeans = KMeans(init="k-means++", n_clusters=int(5))
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

    def insert_vertex(self, v, s):
        pass
    
    def remove_vertex(self, v):
        pass
        
    def translate_vertex(self, coords):
        pass
   
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
    s.set_neighbor(t)
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
