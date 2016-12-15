#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "particles.h"
#include "nn.h"
#include "md3dsystem.h"

void Interactions::init(MDSystem *s){
	sys = s;
	rCutMax = pow(2.0, 1.0 / 6.0);
	kWall = 100;
	bondForces = false;
	feaForces = false;
	gravityForce = false;
	gravity = 0;
	nu = 0.3;
	E = 0;
	kBond = 0;
	kWall = 10;
	kArea = 0;
	setElasticConstants();
	real eps[] = { 1.0 };
	real rCut[] = { pow(2.0, 1.0 / 6.0) };
	real uShift[] = { 1.0 };
	setPairForce(eps, rCut, uShift);

}

void Interactions::reset(){
	rCut.clear();
	uShift.clear();
	epsilon.clear();
	if (sys->DEV == GPU){
		cudaFree(d_epsilon);
		cudaFree(d_rCut);
		cudaFree(d_uShift);
	}
}

void Interactions::setElasticConstants(){
	lambda = E*nu / ((1 - 2 * nu)*(1 + nu));
	mu = E / (2 * (1 + nu));
	Cxxxx = lambda + 2 * mu;
	Cxxyy = lambda;
	Cxyxy = mu;

}

void Interactions::setCuda(){
	int numTypes = sys->particles.numTypes;
	int n2 = numTypes*numTypes;
	cudaMalloc((void**)&d_epsilon, n2*sizeof(real));
	cudaMalloc((void**)&d_rCut, numTypes*numTypes*sizeof(real));
	cudaMalloc((void**)&d_uShift, numTypes*numTypes*sizeof(real));

	cudaMemcpy(d_epsilon, epsilon.data, n2*sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rCut, rCut.data, n2*sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uShift, uShift.data, n2*sizeof(real), cudaMemcpyHostToDevice);
}


void Interactions::setPairForce(real *epsA, real *rCutA, real *ushiftA){
	rCutMax = 0;
	int numTypes = sys->particles.numTypes;
	epsilon.init(numTypes);
	rCut.init(numTypes);
	uShift.init(numTypes);
	for (int i = 0; i < numTypes; i++){
		for (int j = 0; j<numTypes; j++){
			epsilon[i][j] = epsA[i*numTypes + j];
			rCut[i][j] = rCutA[i*numTypes + j];
			if (rCut[i][j] > rCutMax) rCutMax = rCut[i][j];
			uShift[i][j] = ushiftA[i*numTypes + j];
		}
	}
}

real reduce(dvector<real> &vv);

void Interactions::calcForces(){
	switch (sys->pbcType){
	case NOPBC:
		calcWallForces();
		break;
	case XPBC:
		calcTopBottomWallForces();
	case XYPBC:
		break;
	default:
		break;
	}
	if(gravityForce) calcGravityForces();
	if (bondForces) calcBondForces();
	if (feaForces) calcFeaForces();
	if (areaForces) calcAreaForces();
	if (sys->useNN)
		calcPairForcesNN();
	else
		calcPairForces();
	if (sys->DEV == GPU){
		sys->potEnergy = reduce(sys->particles.d_enei);
		sys->virial = reduce(sys->particles.d_viri);
	}
	//sys->particles.feaElements.checkAreas(sys->box);
	//printf("%f\n", sys->potEnergy);
}

void Interactions::calcGravityForces(){
	if (gravity > 0){
		if (sys->DEV == GPU){
			computeGravityForcesGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs,
				sys->interactions.gravity, sys->box);
		}
		else
		{
			#pragma omp parallel for
			for (int i = 0; i < sys->N; i++){
				sys->particles.force[i].y -= sys->particles.mass[i] * gravity;
			}
		}
	}
}

void Interactions::calcBondForces(){

	if (sys->DEV == GPU){
		//int bdg = (sys->particles.bonds.numBonds-1 + 256) / 256;
		//int bdb = 256;
		int bdg = sys->dimGrid;
		int bdb = sys->dimBlock;
		computeBondForcesGPUWrapper(bdg, bdb, sys->particles.d_gpuArgs, sys->interactions.E, 
									sys->box, sys->pbcType);
	}
	else
	{
		calcBondForcesCPU();
	}
}

void Interactions::calcBondForcesCPU(){

		//#pragma omp parallel for
	for (int l = 0; l < sys->particles.bonds.numBonds; l++){
		//for (int l = 0; l < 1; l++){
		int i1 = sys->particles.bonds.bondList[2 * l];
		int i2 = sys->particles.bonds.bondList[2 * l + 1];
		real r0 = sys->particles.bonds.bondLength[l];
		int i = i1;
		int j = i2;
		VectorR dr;
		dr.x = sys->particles.pos[i].x - sys->particles.pos[j].x;
		dr.y = sys->particles.pos[i].y - sys->particles.pos[j].y;
		nearestImage(dr, sys->box, sys->pbcType);
		real r = sqrt(dr.x*dr.x + dr.y*dr.y);
		real deltar = r - r0;
		//kBond = E / r0;
		//kBond = 10;
		real kdr = -kBond*deltar / r;
		real fx = kdr*dr.x;
		real fy = kdr*dr.y;
		sys->particles.force[i].x += fx;
		sys->particles.force[i].y += fy;
		sys->particles.force[j].x -= fx;
		sys->particles.force[j].y -= fy;
		real vir = kBond*deltar*deltar;
		real u = 0.5*vir;
		sys->potEnergy += u;
		//sys->virial += vir;
	}
}


void Interactions::calcWallForces(){
	if (sys->DEV == GPU){
		computeWallForcesGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs,
			sys->interactions.kWall, sys->box);
	}
	else
	{
		forceOnWalls.x0 = 0;
		forceOnWalls.x1 = 0;
		forceOnWalls.y0 = 0;
		forceOnWalls.y1 = 0;
		real rc = pow(2.0, 1.0 / 6.0);

		for (int i = 0; i<sys->N; i++){
			real s0 = sys->particles.sigma[i];
			//Top Wall
			if ((sys->box.y - sys->particles.pos[i].y) < s0*rc){ // collision detection
				real yi = sys->box.y - sys->particles.pos[i].y;
				real f;
				real u = ljwallforce(yi, s0, f);
				sys->particles.force[i].y += -f;
				forceOnWalls.y1 -= -f;
				sys->potEnergy += u;
			}
			//Bottom wall
			else if (sys->particles.pos[i].y < s0*rc){
				real yi = sys->particles.pos[i].y;
				real f;
				real u = ljwallforce(yi, s0, f);
				sys->particles.force[i].y += f;
				forceOnWalls.y1 -= f;
				sys->potEnergy += u;
			}
			//Right Wall
			if ((sys->box.x - sys->particles.pos[i].x) < s0*rc){ // collision detection
				real xi = sys->box.x - sys->particles.pos[i].x;
				real f;
				real u = ljwallforce(xi, s0, f);
				sys->particles.force[i].x += -f;
				forceOnWalls.x1 -= -f;
				sys->potEnergy += u;
			}
			//Left wall
			else if (sys->particles.pos[i].x < s0*rc){
				real xi = sys->particles.pos[i].x;
				real f;
				real u = ljwallforce(xi, s0, f);
				sys->particles.force[i].x += f;
				forceOnWalls.x1 -= f;
				sys->potEnergy += u;
			}

		}
		real Ax = sys->box.y;
		real Ay = sys->box.x;
		sys->wallPressure = forceOnWalls.x0 / Ax +
			-forceOnWalls.x1 / Ax +
			forceOnWalls.y0 / Ay +
			-forceOnWalls.y1 / Ay;
		sys->wallPressure /= 4;
	}
}

void Interactions::calcTopBottomWallForces(){
	if (sys->DEV == GPU){
		computeTopBottomWallForcesGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs,
			sys->interactions.kWall, sys->box);
	}
	else
	{
		forceOnWalls.x0 = 0;
		forceOnWalls.x1 = 0;
		forceOnWalls.y0 = 0;
		forceOnWalls.y1 = 0;
		real rc = pow(2.0, 1.0 / 6.0);
		
		for (int i = 0; i<sys->N; i++){
			real s0 = sys->particles.sigma[i];
			//Top Wall
			if ((sys->box.y - sys->particles.pos[i].y) < s0*rc){ // collision detection
				real yi = sys->box.y - sys->particles.pos[i].y;
				real f;
				//real u = ljwallforce(yi, s0, f);
				f = kWall*yi;
				real u = 0.5*kWall*yi*yi;
				sys->particles.force[i].y += -f;
				forceOnWalls.y1 -= -f;
				sys->potEnergy += u;
			}
			//Bottom wall
			else if (sys->particles.pos[i].y < s0*rc){
				real yi = sys->particles.pos[i].y;
				real f;
				//real u = ljwallforce(yi, s0, f);
				f = kWall*yi;
				real u = 0.5*kWall*yi*yi;
				sys->particles.force[i].y += f;
				forceOnWalls.y1 -= f;
				sys->potEnergy += u;
			}
		}
		real Ax = sys->box.y;
		real Ay = sys->box.x;
		sys->wallPressure = 
			forceOnWalls.x0 / Ax +
			-forceOnWalls.x1 / Ax +
			forceOnWalls.y0 / Ay +
			-forceOnWalls.y1 / Ay;
		sys->wallPressure /= 4;
	}
}


//if (m1 != m2 || j2 < j1) {
//if ((m1 != m2 || j2 < j1) && (mol[j1].inChain == -1 || mol[j1].inChain != mol[j2].inChain || abs(j1 - j2) > 1))

/*
Condition
i       j
-1 and -1  interaction
g  and -1  interaction
-1 and  g  interaction
g  and  g  no interaction

*/


void Interactions::calcPairForces(){

	if (sys->DEV == GPU){
		computeForcesAllGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs, 
									d_epsilon, d_rCut, d_uShift, sys->box, sys->pbcType);
	}
	else
	{
		//#pragma omp parallel for
		sys->virial = 0;
		VectorR fij;
		for (int i = 0; i < sys->N; i++){
			//real temp = 0;
			for (int j = 0; j < sys->N; j++){
				int gi = sys->particles.exclusionGroupIdOfPart[i];
				int gj = sys->particles.exclusionGroupIdOfPart[j];
				if ((i != j) && (gi == -1 || gj == -1 || (gi != gj))){
				//if (i != j){
					real vir = 0.0;
					real u = pairForce(i, j, sys->particles.pos.data(), sys->particles.type.data(), sys->particles.sigma.data(),
						epsilon.data, rCut.data, uShift.data, sys->particles.numTypes, sys->box, sys->pbcType, sys->particles.force.data(),
						vir,fij);
					//#pragma omp atomic
					sys->virial += 0.5*vir;
					sys->potEnergy += 0.5*u;
					if (sys->particles.feaElements.clusters.size() > 0){
						int ci = sys->particles.clusterIdOfPart[i];
						int cj = sys->particles.clusterIdOfPart[j];
						VectorR rci, rcj, drcij;
						rci = sys->particles.feaElements.clusters[ci].cmPos;
						rcj = sys->particles.feaElements.clusters[cj].cmPos;
						drcij.x = rci.x - rcj.x;
						drcij.y = rci.y - rcj.y;
						real dr2 = drcij.x*drcij.x + drcij.y*drcij.y;
						sys->clusterVirial += fij.x*drcij.x + fij.y*drcij.y;
					}
					//temp += 0.5*u;
				}
			}
			//printf("%f\n", temp);
		}
	}
}

void Interactions::calcPairForcesNN(){

	if (sys->DEV == GPU){
		computeForcesNNGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs, sys->neighborlist.d_nnGpuArgs,
								d_epsilon, d_rCut, d_uShift, sys->box, sys->pbcType);
	}
	else{
		#pragma omp parallel num_threads(8)
		#pragma omp for
		for (int pi = 0; pi < sys->N; pi++){
			VectorR fij;
			int pj;
			real u;
			for (int j = 0; j < sys->neighborlist.neighborCount[pi]; j++){
				pj = sys->neighborlist.neighborList[pi*sys->neighborlist.maxNeighborsPerParticle + j];
				real vir = 0;
				u = pairForce(pi, pj, sys->particles.pos.data(), sys->particles.type.data(), sys->particles.sigma.data(),
						epsilon.data, rCut.data, uShift.data, sys->particles.numTypes, sys->box, sys->pbcType, sys->particles.force.data(), 
						vir, fij);

				//#pragma omp atomic
				sys->potEnergy += u*real(0.5);
				//#pragma omp atomic
				sys->virial += vir*real(0.5);
				if (sys->particles.feaElements.clusters.size() > 0){
					int ci = sys->particles.clusterIdOfPart[pi];
					int cj = sys->particles.clusterIdOfPart[pj];
					VectorR rci, rcj, drcij;
					rci = sys->particles.feaElements.clusters[ci].cmPos;
					rcj = sys->particles.feaElements.clusters[cj].cmPos;
					drcij.x = rci.x - rcj.x;
					drcij.y = rci.y - rcj.y;
					nearestImage(drcij, sys->box, sys->pbcType);
					//real dr2 = drcij.x*drcij.x + drcij.y*drcij.y;
					sys->clusterVirial += 0.5*(fij.x*drcij.x + fij.y*drcij.y);
					//printf("%d %d %f %f %f\n",ci, cj, dr2, fij, fij*dr2);
				}

			}
		}
		//printf("%f\n", sys->virial);
	}
}

///// Tetra Forces //////////////////////

void Interactions::calcFeaForces() {
	if (sys->DEV == CPU){
		//calcFeaForcesCPU();
		sys->particles.feaElements.offset = 0;
		sys->particles.feaElements.calcClusterProps();
		calcFeaForcesCPU2();
	}
	else{
		int threadsPerBlock = 256;
		int dimGrid = (sys->particles.feaElements.numTetras + threadsPerBlock - 1) / threadsPerBlock;
		int dimBlock = threadsPerBlock;
		computeFeaForcesGPUWrapper(dimGrid, dimBlock, sys->particles.d_gpuArgs,
			sys->interactions.Cxxxx, sys->interactions.Cxxyy, sys->interactions.Cxyxy,
			sys->box, sys->pbcType);
	}

}

void Interactions::calcFeaForcesCPU1() {

	VectorR u[3], r[3], r0[3], f[3], box;
	int pind[3];
	box = sys->box;
	mat3x3 *xm = sys->particles.feaElements.xm.data();
	int offset = sys->particles.feaElements.offset;

	for (int it = 0; it < sys->particles.feaElements.tetras.size(); it++){
		for (int i = 0; i < 3; i++) {
			int pi = sys->particles.feaElements.tetras[it][i];
			pind[i] = pi;
			r[i].x = sys->particles.pos[pi].x;
			r[i].y = sys->particles.pos[pi].y;
			r0[i].x = sys->particles.feaElements.refPos[pi].x;
			r0[i].y = sys->particles.feaElements.refPos[pi].y;
		}

		real dx01 = r[0].x - r[1].x;
		if (dx01 >  sys->box.x / 2.0) r[1].x += sys->box.x;
		if (dx01 < -sys->box.x / 2.0) r[1].x -= sys->box.x;

		real dx02 = r[0].x - r[2].x;
		if (dx02 > sys->box.x / 2.0) r[2].x += sys->box.x;
		if (dx02 < -sys->box.x / 2.0) r[2].x -= sys->box.x;

		real dy01 = r[0].y - r[1].y;
		if (dy01 > sys->box.y / 2.0) r[1].y += sys->box.y;
		if (dy01 < -sys->box.y / 2.0) r[1].y -= sys->box.y;

		real dy02 = r[0].y - r[2].y;
		if (dy02 > sys->box.y / 2.0) r[2].y += sys->box.y;
		if (dy02 < -sys->box.y / 2.0) r[2].y -= sys->box.y;

		//real vol = 0.5*((r[1].x*r[2].y - r[2].x*r[1].y) - (r[2].x*r[0].y-r[0].x*r[2].y) - (r[1].x*r[2].y-r[2].x*r[1].y));
		//real vol = 0.5*((r0[1].x*r0[2].y - r0[2].x*r0[1].y) - (r0[2].x*r0[0].y - r0[0].x*r0[2].y) - (r0[1].x*r0[2].y - r0[2].x*r0[1].y));
		real vol = 0.5*((r0[2].y - r0[0].y)*(r0[1].x - r0[0].x) - (r0[1].y - r0[0].y)*(r0[2].x - r0[0].x));
		real newVol = 0.5*((r[2].y - r[0].y)*(r[1].x - r[0].x) - (r[1].y - r[0].y)*(r[2].x - r[0].x));
		if (vol*newVol < 0)printf("Error in triangle %d %f %f \n", it, newVol, vol);
		vol = fabs(vol);
		//real vol = e->tetVol[itet];
		//xm = sys->particles.feaElements.xm[it];

		for (int i = 0; i < 3; i++) {
			u[i].x = r[i].x - r0[i].x;
			u[i].y = r[i].y - r0[i].y;
		}

		real a[3], b[3];
		for (int i = 0; i < 3; i++) {
			a[i] = b[i] = 0;
		}
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				a[i] = a[i] + xm[it].m[i][j] * u[j].x;
				b[i] = b[i] + xm[it].m[i][j] * u[j].y;
			}
		}

		real eps[2][2];
		eps[0][0] = a[1] + 0.5f*(a[1] * a[1] + b[1] * b[1]);
		eps[1][1] = b[2] + 0.5f*(a[2] * a[2] + b[2] * b[2]);
		eps[0][1] = 0.5*(a[2] + b[1] + a[1] * a[2] + b[1] * b[2]);
		eps[1][0] = eps[0][1];

		real pesum = 0;
		real cxxxx = Cxxxx;
		real cxxyy = Cxxyy;
		real cxyxy = Cxyxy;
		//cxxxx = 1;
		//cxxyy = 0.1;
		//cxyxy = 0.1;
		pesum = cxxxx*(eps[0][0] * eps[0][0] + eps[1][1] * eps[1][1])
			+ 2.0f*cxxyy*(eps[0][0] * eps[1][1]) + 4.0f*cxyxy*(eps[0][1] * eps[0][1]);
		//printf("%f\n", pesum);
		sys->potEnergy += pesum*vol;

		//        calculate the x,y forces on the 4 nodes

		real fxsum = 0.0;
		real fysum = 0.0;

		//do 14 i = 1, 4
		//	j = npick(i)
		VectorR ftot;
		ftot.x = ftot.y = 0;
		for (int i = 0; i < 3; i++) {
			int j = pind[i];
			fxsum = -2.0*cxxxx*eps[0][0] * xm[it].m[1][i] * (1.0 + a[1])
				- 2.0*cxxxx*eps[1][1] * xm[it].m[2][i] * a[2]
				- 2.0*cxxyy*eps[0][0] * xm[it].m[2][i] * a[2]
				- 2.0*cxxyy*eps[1][1] * xm[it].m[1][i] * (1.0 + a[1])
				- 4.0*cxyxy*eps[0][1] * xm[it].m[2][i] * (1.0 + a[1])
				- 4.0*cxyxy*eps[0][1] * xm[it].m[1][i] * a[2];

			fysum = -2.0*cxxxx*eps[0][0] * xm[it].m[1][i] * b[1]
				- 2.0*cxxxx*eps[1][1] * xm[it].m[2][i] * (1.0 + b[2])
				- 2.0*cxxyy*eps[0][0] * xm[it].m[2][i] * (1.0 + b[2])
				- 2.0*cxxyy*eps[1][1] * xm[it].m[1][i] * b[1]
				- 4.0*cxyxy*eps[0][1] * xm[it].m[1][i] * (1.0 + b[2])
				- 4.0*cxyxy*eps[0][1] * xm[it].m[2][i] * b[1];

			ftot.x += fxsum;
			ftot.y += fysum;
			//real vir = fxsum*r[i].x + fysum*r[i].y;
			//real vir = fxsum*sys->particles.pos[j + e->offset].x + fysum*sys->particles.pos[j + e->offset].y;
			//sys->virial += vir;
			//#pragma omp critical
			{
				//#pragma omp atomic
				sys->particles.force[j + offset].x += fxsum*vol;
				//#pragma omp atomic
				sys->particles.force[j + offset].y += fysum*vol;

			}
			//printf("forces %d %f %f\n", itet, fxsum, fysum);
		}
		//printf("total force for tet %d x:%f y:%f\n", itet, ftot.x, ftot.y);
	}
}

void Interactions::calcFeaForcesCPU2() {

	//VectorR u[3], r[3], r0[3], f[3];
	//int pind[3];
	VectorR box = sys->box;
	mat3x3 *xm = sys->particles.feaElements.xm.data();
	int offset = sys->particles.feaElements.offset;

	//#pragma omp parallel num_threads(8)
	//#pragma omp for
	for (int it = 0; it < sys->particles.feaElements.tetras.size(); it++){
		VectorR u[3], r[3], r0[3], f[3];
		int pind[3];
		for (int i = 0; i < 3; i++) {
			int pi = sys->particles.feaElements.tetras[it][i];
			pind[i] = pi;
			r[i].x = sys->particles.pos[pi].x;
			r[i].y = sys->particles.pos[pi].y;
			r0[i].x = sys->particles.feaElements.refPos[pi].x;
			r0[i].y = sys->particles.feaElements.refPos[pi].y;
		}

		real dx01 = r[0].x - r[1].x;
		if (dx01 >  sys->box.x / 2.0) r[1].x += sys->box.x;
		if (dx01 < -sys->box.x / 2.0) r[1].x -= sys->box.x;

		real dx02 = r[0].x - r[2].x;
		if (dx02 > sys->box.x / 2.0) r[2].x += sys->box.x;
		if (dx02 < -sys->box.x / 2.0) r[2].x -= sys->box.x;

		real dy01 = r[0].y - r[1].y;
		if (dy01 > sys->box.y / 2.0) r[1].y += sys->box.y;
		if (dy01 < -sys->box.y / 2.0) r[1].y -= sys->box.y;

		real dy02 = r[0].y - r[2].y;
		if (dy02 > sys->box.y / 2.0) r[2].y += sys->box.y;
		if (dy02 < -sys->box.y / 2.0) r[2].y -= sys->box.y;

		//real vol = 0.5*((r[1].x*r[2].y - r[2].x*r[1].y) - (r[2].x*r[0].y-r[0].x*r[2].y) - (r[1].x*r[2].y-r[2].x*r[1].y));
		//real vol = 0.5*((r0[1].x*r0[2].y - r0[2].x*r0[1].y) - (r0[2].x*r0[0].y - r0[0].x*r0[2].y) - (r0[1].x*r0[2].y - r0[2].x*r0[1].y));
		real vol = 0.5*((r0[2].y - r0[0].y)*(r0[1].x - r0[0].x) - (r0[1].y - r0[0].y)*(r0[2].x - r0[0].x));
		real newVol = 0.5*((r[2].y - r[0].y)*(r[1].x - r[0].x) - (r[1].y - r[0].y)*(r[2].x - r[0].x));
		if (vol*newVol < 0)printf("Error in triangle %d %f %f \n", it, newVol, vol);
		vol = fabs(vol);
		//real vol = e->tetVol[itet];
		//xm = sys->particles.feaElements.xm[it];

		real Dm[2][2];
		real Bm[2][2];
		real Ds[2][2];
		real F[2][2];
		real P[2][2];
		real H[2][2];
		real E[2][2];
		//Agorithm from fendefo.org page 30

		// calculate Dm from ref pos (Xi - X0)
		Dm[0][0] = r0[0].x - r0[2].x;
		Dm[1][0] = r0[0].y - r0[2].y;
		Dm[0][1] = r0[1].x - r0[2].x;
		Dm[1][1] = r0[1].y - r0[2].y;

		//det and volume of undeformed triangle;
		real detDm = Dm[0][0] * Dm[1][1] - Dm[0][1] * Dm[1][0];
		real vol0 = detDm / 2;

		// Bm inverse of Dm Bm = Inverse(Dm)
		Bm[0][0] = Dm[1][1] / detDm;
		Bm[1][0] = -Dm[1][0] / detDm;
		Bm[0][1] = -Dm[0][1] / detDm;
		Bm[1][1] = Dm[0][0] / detDm;

		//calculate Ds from world positions Ds = (xi-x0)
		Ds[0][0] = r[0].x - r[2].x;
		Ds[1][0] = r[0].y - r[2].y;
		Ds[0][1] = r[1].x - r[2].x;
		Ds[1][1] = r[1].y - r[2].y;

		//Deformation gradient F = Ds*Bm
		F[0][0] = Ds[0][0] * Bm[0][0] + Ds[0][1] * Bm[1][0];
		F[1][0] = Ds[1][0] * Bm[0][0] + Ds[1][1] * Bm[1][0];
		F[0][1] = Ds[0][0] * Bm[0][1] + Ds[0][1] * Bm[1][1];
		F[1][1] = Ds[1][0] * Bm[0][1] + Ds[1][1] * Bm[1][1];

		//Green lagrange strain tensor E = 0.5(Transpose(F)*F - 1)
		E[0][0] = 0.5*(F[0][0] * F[0][0] + F[1][0] * F[1][0] - 1);
		E[1][0] = 0.5*(F[0][0] * F[0][1] + F[1][1] * F[1][0]);
		E[0][1] = 0.5*(F[0][0] * F[0][1] + F[1][0] * F[1][1]);
		E[1][1] = 0.5*(F[0][1] * F[0][1] + F[1][1] * F[1][1] - 1);

		//printf("\nE: %f %f %f %f\n", E[0][0], E[1][0], E[0][1], E[1][1]);
		//printf("\F: %f %f %f %f\n", F[0][0], F[1][0], F[0][1], F[1][1]);

		// Piola tensor
		//St Venant-Kirchooff model
		//P(F) = F(2*mu*E + lambda*Tr(E)*I)

		real trE, ltrE, detF, llogJ, pe;
		int svk = 0;
		if (svk){
			trE = (E[0][0] + E[1][1]);
			ltrE = lambda*trE;
			P[0][0] = ltrE * F[0][0] + 2 * mu* (F[0][0] * E[0][0] + F[0][1] * E[1][0]);
			P[1][0] = ltrE * F[1][0] + 2 * mu* (F[1][0] * E[0][0] + F[1][1] * E[1][0]);
			P[0][1] = ltrE * F[0][1] + 2 * mu* (F[0][0] * E[0][1] + F[0][1] * E[1][1]);
			P[1][1] = ltrE * F[1][1] + 2 * mu* (F[1][0] * E[0][1] + F[1][1] * E[1][1]);
			pe = mu*(E[0][0] * E[0][0] + E[1][0] * E[1][0] + E[0][1] * E[0][1] + E[1][1] * E[1][1]) + 0.5*lambda*trE*trE;
		}
		else{
			//NeoHookean
			//
			detF = F[0][0] * F[1][1] - F[0][1] * F[1][0];
			real logJ = log(detF);
			real llogJ = lambda*logJ;
			real invF[2][2];
			invF[0][0] =  F[1][1] / detF;
			invF[1][0] = -F[1][0] / detF;
			invF[0][1] = -F[0][1] / detF;
			invF[1][1] =  F[0][0] / detF;
			P[0][0] = mu*(F[0][0] - invF[0][0]) + llogJ*invF[0][0];
			P[1][0] = mu*(F[1][0] - invF[0][1]) + llogJ*invF[0][1];
			P[0][1] = mu*(F[0][1] - invF[1][0]) + llogJ*invF[1][0];
			P[1][1] = mu*(F[1][1] - invF[1][1]) + llogJ*invF[1][1];
			real I1 = F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[0][1] * F[0][1] + F[1][1] * F[1][1];
			pe = 0.5*mu*(I1 - 2) - mu*logJ + 0.5*lambda*logJ*logJ;
		}

		//Force calculation
		H[0][0] = -vol0*(P[0][0] * Bm[0][0] + P[0][1] * Bm[0][1]);
		H[1][0] = -vol0*(P[1][0] * Bm[0][0] + P[1][1] * Bm[0][1]);
		H[0][1] = -vol0*(P[0][0] * Bm[1][0] + P[0][1] * Bm[1][1]);
		H[1][1] = -vol0*(P[1][0] * Bm[1][0] + P[1][1] * Bm[1][1]);
		
		//        calculate the x,y forces on the 3 nodes
		//VectorR f[3];
		f[0].x = H[0][0];
		f[0].y = H[1][0];
		f[1].x = H[0][1];
		f[1].y = H[1][1];
		f[2].x = -f[0].x - f[1].x;
		f[2].y = -f[0].y - f[1].y;
		
		for (int i = 0; i < 3; i++){
			int j = pind[i];
			//#pragma omp atomic
			sys->particles.force[j + offset].x += f[i].x;
			//#pragma omp atomic
			sys->particles.force[j + offset].y += f[i].y;
			//#pragma omp atomic
			sys->feaVirial += f[i].x*sys->particles.feaElements.unfoldedPos[j + offset].x +
								f[i].y*sys->particles.feaElements.unfoldedPos[j + offset].y;

		}
		//printf("E: %f\n", pe);
		sys->potEnergy += pe*vol0;

	}
}

void Interactions::calcAreaForces(){
	calcAreaForcesCPU();
}


void Interactions::calcAreaForcesCPU(){

	VectorR r[3], r0[3], f[3], box;
	box = sys->box;

	for (int it = 0; it < sys->particles.feaElements.tetras.size(); it++){
		for (int i = 0; i < 3; i++) {
			int pi = sys->particles.feaElements.tetras[it][i];
			r[i].x = sys->particles.pos[pi].x;
			r[i].y = sys->particles.pos[pi].y;
			r0[i].x = sys->particles.feaElements.refPos[pi].x;
			r0[i].y = sys->particles.feaElements.refPos[pi].y;
		}

		real dx01 = r[0].x - r[1].x;
		if (dx01 > box.x / 2.0) r[1].x += box.x;
		if (dx01 < -box.x / 2.0) r[1].x -= box.x;

		real dx02 = r[0].x - r[2].x;
		if (dx02 > box.x / 2.0) r[2].x += box.x;
		if (dx02 < -box.x / 2.0) r[2].x -= box.x;

		real dy01 = r[0].y - r[1].y;
		if (dy01 > box.y / 2.0) r[1].y += box.y;
		if (dy01 < -box.y / 2.0) r[1].y -= box.y;

		real dy02 = r[0].y - r[2].y;
		if (dy02 > box.y / 2.0) r[2].y += box.y;
		if (dy02 < -box.y / 2.0) r[2].y -= box.y;

		//real vol = 0.5*((r[1].x*r[2].y - r[2].x*r[1].y) - (r[2].x*r[0].y-r[0].x*r[2].y) - (r[1].x*r[2].y-r[2].x*r[1].y));
		//real vol = 0.5*((r0[1].x*r0[2].y - r0[2].x*r0[1].y) - (r0[2].x*r0[0].y - r0[0].x*r0[2].y) - (r0[1].x*r0[2].y - r0[2].x*r0[1].y));
		real A0 = 0.5*((r0[2].y - r0[0].y)*(r0[1].x - r0[0].x) - (r0[1].y - r0[0].y)*(r0[2].x - r0[0].x));
		real A = 0.5*((  r[2].y -  r[0].y)*( r[1].x -  r[0].x) - (r[1].y  -  r[0].y)*( r[2].x -  r[0].x));
		if (A*A0 < 0)printf("Error in triangle %d %f %f \n", it, A, A0);

		//real kArea = 0.001;
		real c = -0.5*kArea*(A - A0) / A0;
		sys->potEnergy += 0.5*kArea*(A - A0)*(A - A0);
		f[1].x =  c*(r[2].y - r[0].y);
		f[1].y = -c*(r[2].x - r[0].x);
		f[2].x = -c*(r[1].y - r[0].y);
		f[2].y =  c*(r[1].x - r[0].x);
		f[0].x = -f[1].x - f[2].x;
		f[0].y = -f[1].y - f[2].y;
		//f[0].x = c*(r[1].y - r[2].y);
		//f[0].y = c*(r[2].x - r[1].x);

		for (int i = 0; i < 3; i++) {
			int pi = sys->particles.feaElements.tetras[it][i];
			sys->particles.pos[pi].x += f[i].x;
			sys->particles.pos[pi].y += f[i].y;
		}
	}
}


