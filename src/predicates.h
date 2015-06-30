#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define REAL double                      /* float or double */
#define REALPRINT doubleprint
#define REALRAND doublerand
#define NARROWRAND narrowdoublerand
#define UNIFORMRAND uniformdoublerand

#define INEXACT                          /* Nothing */
/* #define INEXACT volatile */

extern "C" {

REAL orient2d(REAL *pa, REAL *pb, REAL *pc);
REAL orient3d(REAL *pa, REAL *pb, REAL *pc, REAL *pd);

REAL incircle(REAL *pa, REAL *pb, REAL *pc, REAL *pd);
REAL insphere(REAL *pa, REAL *pb, REAL *pc, REAL *pd);

void exactinit(void);

}
