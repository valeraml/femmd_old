#ifndef __INTER_H__
#define __INTER_H__

#include "def.h"

class MDSystem;

class Interactions{
public:
	MDSystem *sys;

	bool bondForces;
	bool feaForces;
	bool areaForces;
	bool gravityForce;

	ForceOnWalls forceOnWalls;
	real kWall;
	real kBond;
	real kArea;
	real gravity;
	real nu; //poisson ratio
	real E; //Youngs module
	real Cxxxx, Cxxyy, Cxyxy;
	real lambda, mu;


	real rCutMax;
	matNxN rCut;
	matNxN uShift;
	matNxN epsilon;
	real *d_epsilon;
	real *d_rCut;
	real *d_uShift;

	void init(MDSystem *sys);
	void reset();
	void setElasticConstants();
	void setCuda();

	void calcForces();
	void calcGravityForces();
	void calcWallForces();
	void calcTopBottomWallForces();
	void calcBondForces();
	void calcBondForcesCPU();
	void calcAreaForcesCPU();
	void calcAreaForces();
	void calcFeaForces();
	//void calcFeaForcesCPU();
	void calcFeaForcesCPU1();
	void calcFeaForcesCPU2();
	//void calcElementForce(Elastomer* e, int itet);
	void calcPairForces();
	void calcPairForcesNN();
	void setPairForce(real *epsA, real *rCutA, real *ushiftA);

};

real ljwallforce(real zi, real sigma, real &fz);

real ljforce(int i, int j, real dr2, real dx, real dy, real eps, real sigma, real uShift, real &vir);

real pairForce(int i, int j, VectorR* pos, int* type, real *sigma, real *epsilon, real *rCut, 
				real *uShift, int numTypes, VectorR box, PBCTYPE pbcType, VectorR *force, real &vir, VectorR &fij);

void computeForcesAllGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	VectorR box,
	PBCTYPE pbcType);

void computeForcesNNGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	NNGpuArgs *nnArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	VectorR box,
	PBCTYPE pbcType);

void computeBondForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kBond,
	VectorR box,
	PBCTYPE pbcType);

void computeFeaForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real Cxxxx,
	real Cxxyy,
	real Cxyxy,
	VectorR box,
	PBCTYPE pbcType);

void computeWallForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kWall,
	VectorR box);

void computeTopBottomWallForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kWall,
	VectorR box);

void computeGravityForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kWall,
	VectorR box);

#endif