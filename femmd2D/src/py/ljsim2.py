from __future__ import print_function
from numpy import *
from matplotlib.pyplot import *
import sys, os
import pickle 
import random
from md3dsys import *
#import md3dsys


dir = os.getcwd()
print(dir)
os.chdir('..\data')

# VTF filename 
vtffilename = "ljsim2.vtf"
# DATA filename 
datafilename = "ljsim.dat"

# check whether vtf file already exists
print("Creating {}...".format(vtffilename))
# create a new file and write the structure
vtffile = open(vtffilename, 'w')


# SYSTEM CONSTANTS
# density
density = 0.01
numTypes = 1
# timestep
dt = 0.01
# length of run
tmax = 10.0
# number of particles per side for cubic setup
n = 8
T = 1.0
v0 = 0.5

# COMPUTED CONSTANTS
# total number of particles
N = n*n*n
# volume of the system
volume = N/density
# side length of the system
L = volume**(1./3.)

l = L/(3*n)

#pos, vel = createSystem(0,L,L,L,numTypes,T)
createSystem(L,L,L,numTypes,T)

count = 0
for i in range(n):
	for j in range(n):
		for k in range(n):
			#x[:,count] = [i*l, j*l, k*l]
			x1 = i*l+.5
			y1 = j*l+.5
			z1 = k*l+.5
			vx = (random.random()-0.5)*v0
			vy = (random.random()-0.5)*v0
			vz = (random.random()-0.5)*v0
			#addParticle1(x1,y1,z1,vx,vy,vz,1.0,1.0,0)
			#count += 1

vert = loadtxt("nodes_orig.dat")
vert = np.delete(vert,0,1)
#vert = 1000*vert;
np = vert.shape[0]
print(vert)
print(np)
print("#######################")
tetlist = loadtxt("tetras_orig.dat",dtype='int32') # -1 from mathematica meshes
ntet = tetlist.shape[0]
print(tetlist)
print(ntet)
			
addElastomer(vert, tetlist, 0.1,0.1,0.1,1,0)
#addElastomer(vert, tetlist, 2*L/3,0*L/3,2*L/3,1,0)
#addElastomer(vert, tetlist, 1*L/3,0*L/3,2*L/3,1,0)
#addElastomer(vert, tetlist, 0*L/3,1*L/3,2*L/3,1,0)
#addElastomer(vert, tetlist, 2*L/3, 2*L/3,2*L/3,1,0)
#addElastomer(vert, tetlist, 2*L/3, 2*L/3,2*L/3,1,0)
N=getNumberOfParticles()
setSpringConstant(0.25)
print(N)

eps = array([[1.0]])
rc = array([[pow(2.0,1.0/6.0)]])
ushift = array([[1.0]])
setPairForce(eps,rc,ushift)
createNNList()

pos, vel = getPointers()
bonds = getBondsPointer()
nb = bonds.shape[0]
print(pos.shape)
print(bonds.ndim, bonds.shape, nb)
#setVelv0(v0)

zeroStuff()
calcForces()
doStep()
sysOutput()
print(pos[0,0])
print(vel[0,0])

# write the structure of the system into the file: 
# N particles ("atoms") with a radius of 0.5
#vtffile.write('atom 0:{} radius 0.5\n'.format(N-1))
#count=1

#vtffile.write('atom 0:{} type A radius 0.5\n'.format(count-1))
vtffile.write('atom {}:{} type B radius 0.5\n'.format(count,N-1))
for i in range(nb):
	vtffile.write( 'bond {}:{}\n'.format(bonds[i,0],bonds[i,1]) )
vtffile.write('pbc {} {} {}\n'.format(L, L, L))

# write out that a new timestep starts
vtffile.write('timestep\n')
# write out the coordinates of the particles
c=1000
for i in range(N):
    vtffile.write("{} {} {}\n".format(c*pos[i,0], c*pos[i,1], c*pos[i,2]))

	
	
for i in range(0):
	run(100)
	calcProps();
	sysOutput()
    # write out that a new timestep starts
	vtffile.write('timestep\n')
    # write out the coordinates of the particles
	for i in range(N):
		vtffile.write("{} {} {}\n".format(c*pos[i,0], c*pos[i,1], c*pos[i,2]))
			
# close vtf file
print("Closing {}.".format(vtffilename))
vtffile.close()

#main()
