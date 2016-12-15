

#include "device_functions.h"
#include "cuda_runtime.h"
#include "def.h"
#include "particles.h"
#include "nn.h"
#include "CudaUtil.h"


#ifdef USE_DOUBLE
__device__ static double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

#define atomicAdd atomicAdd_double

#endif


__host__ __device__ real ljwallforce(real zi, real sigma, real &fz){
	zi = sigma / zi;
	real zi3 = zi*zi*zi;
	real zi6 = zi3*zi3*zi3;
	real u = real(4.0)*zi6 * (zi6 - real(1.0)) + real(1.0);
	real fcVal = real(48.0) * zi6 * (zi6 - real(0.5))/ zi;
	fz = fabs(fcVal);
	return u;

}


__host__ __device__ real ljforce(real dr2, real dr, real dx, real dy, real eps, real sigma, real uShift, VectorR *f, real &vir){
	real c = real(48.0) * eps / (sigma*sigma);
	real dr2i = sigma*sigma / dr2;
	real dr6i = dr2i*dr2i*dr2i;
	real fcVal = c*dr6i*(dr6i - real(0.5))*dr2i;

	f->x = fcVal*dx;
	f->y = fcVal*dy;
	vir = fcVal*dr2;

	real u = real(4.0) * eps*dr6i*(dr6i - real(1.0)) + uShift;
	return u;
}

__host__ __device__ real pairForce(
	int i,
	int j,
	VectorR* pos,
	int* type,
	real *sigma,
	real *epsilon,
	real *rCut,
	real *uShift,
	int numTypes,
	VectorR box,
	PBCTYPE pbcType,
	VectorR *force,
	real &vir,
	VectorR &fij
){

	real u = 0;
	VectorR f;
	VectorR dr;
	int typei = type[i];
	int typej = type[j];
	real eij = epsilon[typei*numTypes + typej];
	real rCutij = rCut[typei*numTypes + typej];
	real uShiftij = uShift[typei*numTypes + typej];
	real sigmaij = (sigma[i] + sigma[j]) / 2;

	dr.x = pos[i].x - pos[j].x;
	dr.y = pos[i].y - pos[j].y;
	nearestImage(dr, box, pbcType);
	real dr2 = dr.x*dr.x + dr.y*dr.y;
	real rCut2 = rCutij*rCutij*sigmaij*sigmaij;

	if (dr2 < rCut2){
		real dr1 = sqrt(dr2);
		u = ljforce(dr2, dr1, dr.x, dr.y, eij, sigmaij, uShiftij, &f,vir);
		force[i].x += f.x;
		force[i].y += f.y;
		fij = f;
		//force[j].x -= f.x;
		//force[j].y -= f.y;
	}
	return u;
}

///// Pair force all pairs ///////////////////////

__global__ void computeForcesAllGPU(ParticlesGpuArgs *args, 
									real *d_epsilon, 
									real *d_rCut,
									real *d_uShift,
									VectorR box,
									PBCTYPE pbcType)
{
	VectorR *d_pos = args->pos;
	VectorR *d_force = args->force;
	int *d_type = args->type;
	real *d_sigma = args->sigma;
	int numTypes = args->numTypes;
	int n_particles = args->N;
	real *enei = args->d_enei;
	real *viri = args->d_viri;
	VectorR fij;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_particles) return;
	real u = 0;
	real vir = 0;
	for (int j = 0; j<n_particles; j++){
		vir = 0;
		int gi = args->exclusionGroupIdOfPart[i];
		int gj = args->exclusionGroupIdOfPart[j];
		if ((i != j) && (gi == -1 || gj == -1 || (gi != gj))){
		//if (i != j){
			u = pairForce(i, j, d_pos, d_type, d_sigma, d_epsilon, d_rCut, d_uShift, numTypes, box, pbcType, d_force, vir,fij);
			enei[i] += real(0.5)*u;
			viri[i] += real(0.5)*vir;
		}
	}
}

void computeForcesAllGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	VectorR box,
	PBCTYPE pbcType)
{
	computeForcesAllGPU << <dimGrid, dimBlock >> >(partArgs, d_epsilon, d_rCut, d_uShift, box, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


/////// Pair force neighbor list ////////////////////

__global__ void computeForcesNNGPU( ParticlesGpuArgs *partArgs,
									NNGpuArgs *nnArgs,
									real *d_epsilon,
									real *d_rCut,
									real *d_uShift,
									VectorR box,
									PBCTYPE pbcType)
{
	VectorR *d_pos = partArgs->pos;
	real *d_sigma = partArgs->sigma;
	VectorR *d_force = partArgs->force;
	real *enei = partArgs->d_enei;
	real *viri = partArgs->d_viri;
	int *d_type = partArgs->type;
	int n_particles = partArgs->N;
	int numTypes = partArgs->numTypes;
	real vir = 0;
	VectorR fij;

	int *d_nnCount = nnArgs->pNeighborCount;
	int *d_nn = nnArgs->pNeighborList;
	int maxNeighborsPerParticle = nnArgs->maxNeighborsPerParticle;

	int pi = threadIdx.x + blockIdx.x * blockDim.x;
	if (pi >= n_particles) return;
	real u;
	for (int j = 0; j<d_nnCount[pi]; j++){
		vir = 0;
		int pj = d_nn[pi*maxNeighborsPerParticle + j];
		u = pairForce(pi, pj, d_pos, d_type, d_sigma, d_epsilon, d_rCut, d_uShift, numTypes, box, pbcType, d_force, vir,fij);
		enei[pi] += real(0.5)*u;
		viri[pi] += real(0.5)*vir;
	}
}


void computeForcesNNGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	NNGpuArgs *nnArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	VectorR box,
	PBCTYPE pbcType)
{
	computeForcesNNGPU <<<dimGrid, dimBlock >>>(partArgs, nnArgs, d_epsilon, d_rCut, d_uShift, box, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

/////// Bond forces /////////////////////////


__global__ void computeBondForcesGPU(ParticlesGpuArgs *partArgs, real E, VectorR box, PBCTYPE pbcType){
	
	int n_particles = partArgs->N;
	int *bondCountNN = partArgs->boundCountNN;
	int maxBondPartnersNN = partArgs->maxBondPartnersNN;
	int *bondListNN = partArgs->bondListNN;
	real *bondLengthNN = partArgs->bondLengthNN;
	VectorR *pos = partArgs->pos;
	VectorR *force = partArgs->force;
	real *enei = partArgs->d_enei;
	real *viri = partArgs->d_viri;
	real u, r0, r, deltar, kdr, fx, fy;
	VectorR dr;

	int pi = threadIdx.x + blockIdx.x * blockDim.x;
	real kBond;

	if (pi >= n_particles) return;
	enei[pi] = 0;
	for (int j = 0; j < bondCountNN[pi]; j++){
		int pj = bondListNN[pi*maxBondPartnersNN + j];
		r0 = bondLengthNN[pi*maxBondPartnersNN + j];

		dr.x = pos[pi].x - pos[pj].x;
		dr.y = pos[pi].y - pos[pj].y;
		nearestImage(dr, box, pbcType);

		r = sqrt(dr.x*dr.x + dr.y*dr.y);
		deltar = r - r0;
		kBond = E / r0;
		kdr = -kBond*deltar / r;
		fx = kdr*dr.x;
		fy = kdr*dr.y;
		force[pi].x += fx;
		force[pi].y += fy;
		u = 0.5*kBond*deltar*deltar;
		enei[pi] += real(0.5)*u;
		//viri[pi] += real(0.5)*kdr*2*u;
	}
	//if(enei[pi] > 0)printf("%f\n", enei[pi]);
}

__global__ void computeBondForcesGPU1(ParticlesGpuArgs *partArgs, real E, VectorR box, PBCTYPE pbcType)
{
	VectorR *pos = partArgs->pos;
	VectorR *force = partArgs->force;
	int numBonds = partArgs->numBonds;
	int *bondList = partArgs->bondList;
	real *bondLength = partArgs->bondLength;
	real *enei = partArgs->d_enei;
	real *viri = partArgs->d_viri;
	int pi, pj, l1;
	real r0, r, deltar, kdr, fx, fy, u, vir;
	VectorR dr;

	l1 = threadIdx.x + blockIdx.x * blockDim.x;
	if (l1 >= numBonds)return;
	int l = l1;
	real kBond;
	//for (l = 0; l < numBonds; l++){
		pi = bondList[2 * l];
		pj = bondList[2 * l + 1];
		r0 = bondLength[l];
		dr.x = pos[pi].x - pos[pj].x;
		dr.y = pos[pi].y - pos[pj].y;
		nearestImage(dr, box, pbcType);

		r = sqrt(dr.x*dr.x + dr.y*dr.y);
		deltar = r - r0;
		kBond = E / r0;
		kdr = -kBond*deltar / r;
		fx = kdr*dr.x;
		fy = kdr*dr.y;

		atomicAdd(&force[pi].x,fx);
		atomicAdd(&force[pi].y,fy);
		atomicAdd(&force[pj].x,-fx);
		atomicAdd(&force[pj].y,-fy);
		u = 0.5*kBond*deltar*deltar;
		vir = real(0.5)*kBond*deltar;
		atomicAdd(&enei[pi], real(0.5)*vir);
		atomicAdd(&enei[pj], real(0.5)*vir);
		//atomicAdd(&viri[pi], real(0.5)*vir);
		//atomicAdd(&viri[pj], real(0.5)*vir);

	//}

}

void computeBondForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real E,
	VectorR box,
	PBCTYPE pbcType)
{
	computeBondForcesGPU <<<dimGrid, dimBlock >>>(partArgs, E, box, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

////////Fea Forces /////////////////////

__global__ void computeFeaForcesGPU(ParticlesGpuArgs *partArgs, real Cxxxx, real Cxxyy, real Cxyxy, VectorR box, bool pbc)
{
	VectorR *pos = partArgs->pos;
	VectorR *force = partArgs->force;

	VectorR *refPos = partArgs->refPos;
	tetraIndexes *tetras = partArgs->tetras;
	mat3x3 *xm = partArgs->xm;
	int ntetras = partArgs->ntetras;
	int offset = partArgs->offset;

	real *enei = partArgs->d_enei;
	real *viri = partArgs->d_viri;
	real vir;
	int itet;

	itet = threadIdx.x + blockIdx.x * blockDim.x;
	if (itet >= ntetras)return;

	VectorR u[3], r[3], r0[3];
	real a[3], b[3];
	int pind[3];

	#pragma unroll
	for (int i = 0; i < 3; i++) {
		int pi = tetras[itet].ind[i];
		pind[i] = pi;
		r[i].x = pos[pi + offset].x;
		r[i].y = pos[pi + offset].y;
		r0[i].x = refPos[pi + offset].x;
		r0[i].y = refPos[pi + offset].y;
	}

	real dx01 = r[0].x - r[1].x;
	if (dx01 >  box.x / 2.0) r[1].x += box.x;
	if (dx01 < -box.x / 2.0) r[1].x -= box.x;

	real dx02 = r[0].x - r[2].x;
	if (dx02 >  box.x / 2.0) r[2].x += box.x;
	if (dx02 < -box.x / 2.0) r[2].x -= box.x;

	real dy01 = r[0].y - r[1].y;
	if (dy01 >  box.y / 2.0) r[1].y += box.y;
	if (dy01 < -box.y / 2.0) r[1].y -= box.y;

	real dy02 = r[0].y - r[2].y;
	if (dy02 >  box.y / 2.0) r[2].y += box.y;
	if (dy02 < -box.y / 2.0) r[2].y -= box.y;

	//real tetVol = 0.5*((r[1].x*r[2].y - r[2].x*r[1].y) - (r[2].x*r[0].y-r[0].x*r[2].y) - (r[1].x*r[2].y-r[2].x*r[1].y));
	//real refVol = real(0.5)*((r0[1].x*r0[2].y - r0[2].x*r0[1].y) - (r0[2].x*r0[0].y - r0[0].x*r0[2].y) - (r0[1].x*r0[2].y - r0[2].x*r0[1].y));

	real tetVol = real(0.5)*((r0[2].y - r0[0].y)*(r0[1].x - r0[0].x) - (r0[1].y - r0[0].y)*(r0[2].x - r0[0].x));
	real newVol = real(0.5)*((r[2].y - r[0].y)*(r[1].x - r[0].x) - (r[1].y - r[0].y)*(r[2].x - r[0].x));
	if (newVol*tetVol < 0) printf("Error in triangle %f %f \n", newVol, tetVol);
	tetVol = fabs(tetVol);

	mat3x3 mm;
	mat3x3 aa;
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		//int ip = tetras[itet].ind[j];
		aa.m[i][0] = 1.0;
		aa.m[i][1] = r0[i].x;
		aa.m[i][2] = r0[i].y;

		u[i].x = r[i].x - r0[i].x;
		u[i].y = r[i].y - r0[i].y;
		a[i] = b[i] = real(0.0);
		//printf("pos %d %f %f\n", itet, u[i].x, u[i].y);
	}
	invertMatrix(&aa.m[0][0], &mm.m[0][0]);
	//mm = xm[itet];

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		#pragma unroll
		for (int i = 0; i < 3; i++) {
			//a[i] = a[i] + xm[itet].m[i][j] * u[j].x;
			//b[i] = b[i] + xm[itet].m[i][j] * u[j].y;
			a[i] = a[i] + mm.m[i][j] * u[j].x;
			b[i] = b[i] + mm.m[i][j] * u[j].y;
		}
	}

	real eps[2][2];
	eps[0][0] = a[1] + real(0.5)*(a[1] * a[1] + b[1] * b[1]);
	eps[1][1] = b[2] + real(0.5)*(a[2] * a[2] + b[2] * b[2]);
	eps[0][1] = 0.5*(a[2] + b[1] + a[1] * a[2] + b[1] * b[2]);
	eps[1][0] = eps[0][1];

	real pesum = real(0.0);
	real cxxxx = Cxxxx;
	real cxxyy = Cxxyy;
	real cxyxy = Cxyxy;
	
	pesum = cxxxx*(eps[0][0] * eps[0][0] + eps[1][1] * eps[1][1])
		+ real(2.0)*cxxyy*(eps[0][0] * eps[1][1]) + real(4.0)*cxyxy*(eps[0][1] * eps[0][1]);
	real eneVal = pesum*tetVol;

	real fxsum = real(0.0);
	real fysum = real(0.0);

	VectorR ftot;
	ftot.x = ftot.y = 0;
	#pragma unroll
	for (int i = 0; i < 3; i++) {
		//int j = tetras[itet].ind[i];
		int j = pind[i];
		fxsum = - real(2.0)*cxxxx*eps[0][0] * mm.m[1][i] * (real(1.0) + a[1])
			    - real(2.0)*cxxxx*eps[1][1] * mm.m[2][i] * a[2]
			    - real(2.0)*cxxyy*eps[0][0] * mm.m[2][i] * a[2]
			    - real(2.0)*cxxyy*eps[1][1] * mm.m[1][i] * (real(1.0) + a[1])
			    - real(4.0)*cxyxy*eps[0][1] * mm.m[2][i] * (real(1.0) + a[1])
			    - real(4.0)*cxyxy*eps[0][1] * mm.m[1][i] * a[2];

		fysum = - real(2.0)*cxxxx*eps[0][0] * mm.m[1][i] * b[1]
			    - real(2.0)*cxxxx*eps[1][1] * mm.m[2][i] * (real(1.0) + b[2])
			    - real(2.0)*cxxyy*eps[0][0] * mm.m[2][i] * (real(1.0) + b[2])
			    - real(2.0)*cxxyy*eps[1][1] * mm.m[1][i] * b[1]
			    - real(4.0)*cxyxy*eps[0][1] * mm.m[1][i] * (real(1.0) + b[2])
			    - real(4.0)*cxyxy*eps[0][1] * mm.m[2][i] * b[1];

		ftot.x += fxsum;
		ftot.y += fysum;
		//vir = fxsum*r[i].x + fysum*r[i].y;
		//vir = fxsum*pos[j + offset].x + fysum*pos[j + offset].y;

		atomicAdd(&force[j + offset].x, fxsum*tetVol);
		atomicAdd(&force[j + offset].y, fysum*tetVol);
		atomicAdd(&enei[i+offset], eneVal/3);
		//atomicAdd(&viri[i + offset], vir);

		//force[j + offset].x = force[j + offset].x + fxsum*tetVol;
		//force[j + offset].y = force[j + offset].y + fysum*tetVol;
		//printf("forces %d %f %f\n", itet, fxsum, fysum);	
	}
	//printf("total force for tet %d x:%f y:%f\n", itet, ftot.x, ftot.y);

}

void computeFeaForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real Cxxxx,
	real Cxxyy,
	real Cxyxy,
	VectorR box,
	PBCTYPE pbcType)
{
	//int bdg = (partArgs->numBonds-1 + 256) / 256;
	//int bdb = 32;

	computeFeaForcesGPU << <dimGrid, dimBlock >> >(partArgs, Cxxxx, Cxxyy, Cxyxy, box, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

//////// Wall Forces ////////////////////

__global__ void computeWallForcesGPU(ParticlesGpuArgs *partArgs, real kWall, VectorR box){

	int n_particles = partArgs->N;
	VectorR *pos = partArgs->pos;
	real *radius = partArgs->radius;
	VectorR *force = partArgs->force;
	real *enei = partArgs->d_enei;
	ForceOnWalls forceOnWalls;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_particles) return;
	enei[i] = 0;

	forceOnWalls.x0 = 0;
	forceOnWalls.x1 = 0;
	forceOnWalls.y0 = 0;
	forceOnWalls.y1 = 0;
	real u = 0;
	real l0 = radius[i];
	if (pos[i].x > box.x - l0){
		real deltaX = (pos[i].x - (box.x - l0));
		real f = -kWall*deltaX;
		force[i].x += f;
		forceOnWalls.x0 -= f;
		u += 0.5*kWall*deltaX*deltaX;
	}
	else if (pos[i].x < l0){
		real deltaX = (pos[i].x - l0);
		real f = -kWall*deltaX;
		force[i].x += f;
		forceOnWalls.x1 -= f;
		u += 0.5*kWall*deltaX*deltaX;
	}

	if (pos[i].y > box.y - l0){ // collision detection
		real deltaY = (pos[i].y - (box.y - l0));
		real f = -kWall*deltaY;
		force[i].y += f;
		forceOnWalls.y0 -= f;
		u += 0.5*kWall*deltaY*deltaY;
	}
	else if (pos[i].y < l0){
		real deltaY = (pos[i].y - l0);
		real f = -kWall*deltaY;
		force[i].y += f;
		forceOnWalls.y1 -= f;
		u += 0.5*kWall*deltaY*deltaY;
	}


	real Ax = box.y;
	real Ay = box.x;
	real pressure = 
		forceOnWalls.x0 / Ax +
		-forceOnWalls.x1 / Ax +
		forceOnWalls.y0 / Ay +
		-forceOnWalls.y1 / Ay;
	pressure /= 4;
	enei[i] += u;
}

void computeWallForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kWall,
	VectorR box)
{
	computeWallForcesGPU << <dimGrid, dimBlock >> >(partArgs, kWall, box);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void computeTopBottomWallForcesGPU(ParticlesGpuArgs *partArgs, real kWall, VectorR box){

	int n_particles = partArgs->N;
	VectorR *pos = partArgs->pos;
	real *radius = partArgs->radius;
	VectorR *force = partArgs->force;
	real *enei = partArgs->d_enei;
	ForceOnWalls forceOnWalls;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_particles) return;
	enei[i] = 0;

	forceOnWalls.x0 = 0;
	forceOnWalls.x1 = 0;
	forceOnWalls.y0 = 0;
	forceOnWalls.y1 = 0;
	real u = 0;
	real l0 = radius[i]*1.1;

	if (pos[i].y > box.y - l0){ // collision detection
		real deltaY = (pos[i].y - (box.y - l0));
		real f = -kWall*deltaY;
		force[i].y += f;
		forceOnWalls.y0 -= f;
		u += 0.5*kWall*deltaY*deltaY;
	}
	else if (pos[i].y < l0){
		real deltaY = (pos[i].y - l0);
		real f = -kWall*deltaY;
		force[i].y += f;
		forceOnWalls.y1 -= f;
		u += 0.5*kWall*deltaY*deltaY;
	}
	real Ax = box.y;
	real Ay = box.x;
	real pressure = forceOnWalls.x0 / Ax +
		-forceOnWalls.x1 / Ax +
		forceOnWalls.y0 / Ay +
		-forceOnWalls.y1 / Ay;
	pressure /= 4;
	enei[i] += u;
}

void computeTopBottomWallForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real kWall,
	VectorR box)
{
	computeTopBottomWallForcesGPU << <dimGrid, dimBlock >> >(partArgs, kWall, box);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

__global__ void computeGravityForcesGPU(ParticlesGpuArgs *partArgs, real gravity, VectorR box){

	int n_particles = partArgs->N;
	VectorR *pos = partArgs->pos;
	real *mass = partArgs->mass;
	VectorR *force = partArgs->force;
	real *enei = partArgs->d_enei;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_particles) return;
	enei[i] = 0;
	force[i].y -= mass[i] * gravity;
	enei[i] += mass[i]*gravity*pos[i].y;
}

void computeGravityForcesGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	real gravity,
	VectorR box)
{
	computeGravityForcesGPU << <dimGrid, dimBlock >> >(partArgs, gravity, box);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void computeForcesGPUOnce(ParticlesGpuArgs *partArgs,
	NNGpuArgs *nnArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	real gravity,
	real kBond,
	VectorR box,
	PBCTYPE pbcType)
{
	VectorR *pos = partArgs->pos;
	real *sigma = partArgs->sigma;
	VectorR *force = partArgs->force;
	real *mass = partArgs->mass;
	real *enei = partArgs->d_enei;
	real *viri = partArgs->d_viri;
	int *type = partArgs->type;
	int n_particles = partArgs->N;
	int numTypes = partArgs->numTypes;

	int *bondCountNN = partArgs->boundCountNN;
	int maxBondPartnersNN = partArgs->maxBondPartnersNN;
	int *bondListNN = partArgs->bondListNN;
	real *bondLengthNN = partArgs->bondLengthNN;
	int numBonds = partArgs->numBonds;
	real u, r0, r, deltar, kdr, fx, fy, vir;
	VectorR dr;

	int *nnCount = nnArgs->pNeighborCount;
	int *nn = nnArgs->pNeighborList;
	int maxNeighborsPerParticle = nnArgs->maxNeighborsPerParticle;

	int pi = threadIdx.x + blockIdx.x * blockDim.x;
	if (pi >= n_particles) return;
	enei[pi] = 0;

	/// Gravity Forces
	if (gravity > 0){
		force[pi].y -= mass[pi] * gravity;
		enei[pi] += mass[pi] * gravity*pos[pi].y;
	}

	/// Bonds Forces
	if (numBonds > 0){
		for (int j = 0; j < bondCountNN[pi]; j++){
			int pj = bondListNN[pi*maxBondPartnersNN + j];
			r0 = bondLengthNN[pi*maxBondPartnersNN + j];

			dr.x = pos[pi].x - pos[pj].x;
			dr.y = pos[pi].y - pos[pj].y;
			nearestImage(dr, box, pbcType);
			r = sqrt(dr.x*dr.x + dr.y*dr.y);
			deltar = r - r0;
			kdr = -kBond*deltar / r;
			fx = kdr*dr.x;
			fy = kdr*dr.y;
			force[pi].x += fx;
			force[pi].y += fy;
			u = 0.5*kBond*deltar*deltar;
			enei[pi] += 0.5f*u;
		}
	}

	////Pair Forces 
	VectorR fij;
	for (int j = 0; j<nnCount[pi]; j++){
		int pj = nn[pi*maxNeighborsPerParticle + j];
		u = pairForce(pi, pj, pos, type, sigma, d_epsilon, d_rCut, d_uShift, numTypes, box, pbcType, force,vir,fij);
		enei[pi] += real(0.5)*u;
		viri[pi] += real(0.5)*vir;
	}
}


void computeForcesNNGPUWrapper(
	int dimGrid,
	int dimBlock,
	ParticlesGpuArgs *partArgs,
	NNGpuArgs *nnArgs,
	real *d_epsilon,
	real *d_rCut,
	real *d_uShift,
	real gravity,
	real kBond,
	VectorR box,
	PBCTYPE pbcType)
{
	computeForcesGPUOnce << <dimGrid, dimBlock >> >(partArgs, nnArgs, d_epsilon, d_rCut, d_uShift, gravity, kBond, box, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


/*

template <typename Pot>
__global__ void computeForcesNNGPU(Pot *pot, VectorR *d_pos, VectorR *d_force, int n_particles, int *d_nn, int *d_nnCount,
	int maxNeighborsPerParticle, VectorR box, real rCut2, real *enei){

	int pi = threadIdx.x + blockIdx.x * blockDim.x;
	if (pi >= n_particles) return;
	enei[pi] = 0;
	VectorR fij;
	real eneij;
	VectorR dr;
	int pj;
	for (int j = 0; j<d_nnCount[pi]; j++){
		pj = d_nn[pi*maxNeighborsPerParticle + j];
		dr.x = d_pos[pi].x - d_pos[pj].x;
		dr.y = d_pos[pi].y - d_pos[pj].y;
		dr.z = d_pos[pi].z - d_pos[pj].z;
		nearestImageGPU1(&dr, &box);
		real dr2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
		pot->evalForceCPU(pi, pj, dr2, dr, &fij, &eneij);
		d_force[pi].x += fij.x;
		d_force[pi].y += fij.y;
		d_force[pi].z += fij.z;
		enei[pi] += eneij;
		//		if(dr2 <= rCut2){
		//			computeLJPairForceGPU(dr2, dr, &fij, &eneij);
		//			d_force[pi].x += fij.x;
		//			d_force[pi].y += fij.y;
		//			d_force[pi].z += fij.z;
		//			enei[pi] += eneij;
		//		}
	}
}


//void PairInteractions::computeForcesAll(){
template <typename Pot>
void computeForcesAllGPUWrapper(int dimGrid, int dimBlock, Pot *pot,
VectorR *d_pos, VectorR *d_force, int n_particles, VectorR box, real rCut2, real *ene){
real *d_enei, *h_enei;
cudaMallocHost((void**)&h_enei, n_particles*sizeof(real));
cudaMalloc((void**)&d_enei, n_particles*sizeof(real));
computeForcesAllGPU << <dimGrid, dimBlock >> >(pot, d_pos, d_force, n_particles, box, rCut2, d_enei);
HANDLE_ERROR(cudaDeviceSynchronize());
cudaMemcpy(h_enei, d_enei, n_particles*sizeof(real), cudaMemcpyDeviceToHost);
*ene = 0;
for (int i = 0; i<n_particles; i++){ *ene += h_enei[i]; }
*ene *= .5;
cudaFree(d_enei);
cudaFree(h_enei);
}

template void computeForcesAllGPUWrapper<LJPotential>(int g, int bl, LJPotential *p,
VectorR *r, VectorR *f, int n, VectorR b, real r2, real *e);

//__global__ void computeForcesNNGPU(VectorR *d_pos, VectorR *d_force, int *d_nn, int *d_nnCount,
//									int maxNeighborsPerParticle, VectorR box, real *enei)
template <typename Pot>
void computeForcesNNGPUWrapper(int dimGrid, int dimBlock, Pot *pot,
VectorR *d_pos, VectorR *d_force, int n_particles,
int *d_nn, int *d_nnCount, int maxNeighborsPerParticle,
VectorR box, real rCut2, real *ene){
real *d_enei, *h_enei;
//FIXME allocate this outside this function
cudaMallocHost((void**)&h_enei, n_particles*sizeof(real));
cudaMalloc((void**)&d_enei, n_particles*sizeof(real));
computeForcesNNGPU << <dimGrid, dimBlock >> >(pot, d_pos, d_force, n_particles, d_nn, d_nnCount, maxNeighborsPerParticle, box, rCut2, d_enei);
HANDLE_ERROR(cudaDeviceSynchronize());
cudaMemcpy(h_enei, d_enei, n_particles*sizeof(real), cudaMemcpyDeviceToHost);
*ene = 0;
for (int i = 0; i<n_particles; i++){ *ene += h_enei[i]; }
*ene *= .5;
cudaFree(d_enei);
cudaFreeHost(h_enei);
}

template void computeForcesNNGPUWrapper<LJPotential>(int g, int bl, LJPotential *p,
VectorR *r, VectorR *f, int n, int *nn, int *nnCount, int m, VectorR b, real r2, real *e);

*/