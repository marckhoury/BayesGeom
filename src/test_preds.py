from predicates import orient2d, incircle, orient3d, insphere

import numpy as np

pa = np.array([0, 0], dtype=np.float64)
pb = np.array([0, 1], dtype=np.float64)
pc = np.array([0.5, 0.5], dtype=np.float64)
pd = np.array([-0.5, 0.5], dtype=np.float64)

print orient2d(pa, pb, pc)
print orient2d(pa, pc, pb)

print incircle(pa, pb, pc, pd)
print incircle(pa, pb, pd, pc)

pa = np.array([0, 0, 0], dtype=np.float64)
pb = np.array([0, 1, 0], dtype=np.float64)
pc = np.array([0, 0, 1], dtype=np.float64)
pd = np.array([10, 0, 0], dtype=np.float64)

print orient3d(pa, pb, pc, pd)
print orient3d(pa, pb, pd, pc)

print insphere(pa, pb, pc, pd)
print insphere(pa, pb, pd, pc)


