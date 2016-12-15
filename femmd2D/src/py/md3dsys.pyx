import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
from cython cimport view

ctypedef float real

cdef extern from "interface.h":
	# getters

	cdef real* c_getPosPtr()
	cdef real* c_getVelPtr()
	cdef int* c_getBondsPtr()
	cdef int c_getNumberOfParticles()
	cdef int c_getNumberOfBonds()

	# setters

	cdef void c_setPairForce(real *e, real *rc, real *us)
	cdef void c_setPos(real *r)
	cdef void c_setVelv0(real v0)
	cdef void c_setSpringConstant(real k)

	# others

	cdef void c_createSystem(real Lx, real Ly, real Lz, int numTypes, real T)
	cdef void c_createNNList()
	cdef void c_addParticle(real x, real y, real z, real m, real sig, int typ)
	cdef void c_addParticle1(real x, real y, real z, real vx, real vy, real vz, real m, real sig, int typ)
	cdef void c_addElastomer(real *v, int numVert, int *tet, int numTet, real x, real y, real z, int typ, int g)
	cdef void c_zeroStuff()
	cdef void c_calcForces()
	cdef void c_doStep()
	cdef void c_calcProps()
	cdef void c_run(int steps)
	cdef void c_sysOutput()
#	cdef void c_main()
	
'''
# getters

cdef extern real* c_getPosPtr()
cdef extern real* c_getVelPtr()
cdef extern int* c_getBondsPtr()
cdef extern int c_getNumberOfParticles()
cdef extern int c_getNumberOfBonds()

# setters

cdef extern void c_setPairForce(real *e, real *rc, real *us)
cdef extern void c_setPos(real *r)
cdef extern void c_setVelv0(real v0)
cdef extern void c_setSpringConstant(real k)

# others

cdef extern void c_createSystem(real Lx, real Ly, real Lz, int numTypes, real T)
cdef extern void c_createNNList()
cdef extern void c_addParticle(real x, real y, real z, real m, real sig, int typ)
cdef extern void c_addParticle1(real x, real y, real z, real vx, real vy, real vz, real m, real sig, int typ)
cdef extern void c_addElastomer(real *v, int numVert, int *tet, int numTet, real x, real y, real z, int typ, int g)
cdef extern void c_zeroStuff()
cdef extern void c_calcForces()
cdef extern void c_doStep()
cdef extern void c_calcProps()
cdef extern void c_run(int steps)
cdef extern void c_sysOutput()
cdef extern void c_main()

'''


# getters

def getPointers():
	N = c_getNumberOfParticles();
	size = 3*N
	#pos = <real[:size]> c_getPosPtr()
	#vel = <real[:size]> c_getVelPtr()
	pos = <real[:N,:3]> c_getPosPtr()
	vel = <real[:N,:3]> c_getVelPtr()
	return pos,vel

def getBondsPointer():
	nb = c_getNumberOfBonds();
	bonds = <int[:nb,:2]> c_getBondsPtr();
	return bonds

def getNumberOfParticles():
	return c_getNumberOfParticles()

# setters

def setPairForce(np.ndarray[real, ndim=2, mode='c'] eps not None, np.ndarray[real, ndim=2, mode='c'] rcut not None,np.ndarray[real, ndim=2, mode='c'] ushift not None):
	c_setPairForce(<real*>eps.data,<real*>rcut.data,<real*>ushift.data)

def setPos(np.ndarray[real, ndim=1, mode='c'] r not None):
	c_setPos(&r[0])
	
def setVelv0(real v0):
	c_setVelv0(v0)

def setSpringConstant(real k):
	c_setSpringConstant(k)

# others


def createSystem(real Lx, real Ly, real Lz, int numTypes, real T):
	c_createSystem(Lx,Ly,Lz,numTypes,T)

def createNNList():
	c_createNNList()
	
def addParticle(real x, real y, real z, real m, real sig, int typ):
	c_addParticle(x,y,z,m,sig,typ)
	
def addParticle1(real x, real y, real z, real vx, real vy, real vz, real m, real sig, int typ):
	c_addParticle1(x,y,z,vx,vy,vz,m,sig,typ)

def addElastomer(np.ndarray[real, ndim=2, mode='c'] vertices not None,
				 np.ndarray[int, ndim=2, mode='c'] tetras not None,
				 real x, real y, real z, int typ, int g):
	numVert = vertices.shape[0]
	numTet = tetras.shape[0]
	#c_addElastomer(<real*>vertices,numVert,<int*>tetras,numTet,x,y,z,typ,g)
	c_addElastomer(&vertices[0,0],numVert,&tetras[0,0],numTet,x,y,z,typ,g)
	
def zeroStuff():
	c_zeroStuff()
	
def calcForces():
	c_calcForces()
	
def calcProps():
	c_calcProps()
	
def doStep():
	c_doStep()
	
def run(int steps):
	c_run(steps)
	
def sysOutput():
	c_sysOutput()

#def main():
#	c_main()
