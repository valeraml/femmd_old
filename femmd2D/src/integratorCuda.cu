

#include "def.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "CudaUtil.h"
#include "particles.h"

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

__device__ static double atomicMax(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(  ::fmax(val,__longlong_as_double(assumed)) ));
	} while (assumed != old);
	return __longlong_as_double(old);
}

#define atomicAdd atomicAdd_double

#else

__device__ static float atomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

#endif

__device__ real randonNumber(int ind, curandState* globalState){
	curandState localState = globalState[ind];
	//float RANDOM = curand_uniform(&localState);
	float RANDOM = curand_normal(&localState);
	globalState[ind] = localState;
	return RANDOM;
}


/*
__host__ __device__ void applyBoundaryCondition(VectorR &r, VectorR &box, PBCTYPE pbcType){
	switch (pbcType){
	case XYZPBC:
		pbcCalc(r, box, x);
		pbcCalc(r, box, y);
		pbcCalc(r, box, z);
		break;
	case XYPBC:
		pbcCalc(r, box, x);
		pbcCalc(r, box, y);
		break;
	case XPBC:
		pbcCalc(r, box, x);
		break;
	case NOPBC:
		break;
	}
}
*/

__global__ void rescaleVelocitiesGPU(VectorR *v, real factor, int n_particles){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < n_particles){
		v[i].x *= factor;
		v[i].y *= factor;
		i += blockDim.x * gridDim.x;
	}
}


__global__ void zeroGPU(ParticlesGpuArgs *args){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < args->N){
		args->acc[i].x = 0;
		args->acc[i].y = 0;
		args->force[i].x = 0;
		args->force[i].y = 0;
		args->d_enei[i] = 0;
		args->d_viri[i] = 0;
		i += blockDim.x * gridDim.x;
	}
}

__device__ real d_tempKinEneVVMax[2];

__global__ void integratorStep1GPU(ParticlesGpuArgs *args, VectorR box, real dt, PBCTYPE pbcType){
	// Perform LeapFrog step 1, boundary condition and zero values
	VectorR *pos, *vel, *acc, *force;
	pos = args->pos;
	vel = args->vel;
	acc = args->acc;
	force = args->force;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < args->N){

		vel[i].x = vel[i].x + real(0.5)*dt*acc[i].x;
		vel[i].y = vel[i].y + real(0.5)*dt*acc[i].y;

		pos[i].x = pos[i].x + dt*vel[i].x;
		pos[i].y = pos[i].y + dt*vel[i].y;
		
		//if (pbcType == XYZPBC){
		//	pbcCalc(pos[i], box, x);
		//	pbcCalc(pos[i], box, y);
		//	pbcCalc(pos[i], box, z);
		//}
		applyBoundaryCondition(pos[i], box, pbcType);
		
		//zero stuff
		acc[i].x = 0;
		acc[i].y = 0;
		force[i].x = 0;
		force[i].y = 0;
		args->d_enei[i] = 0;
		args->d_viri[i] = 0;
		
		i += blockDim.x * gridDim.x;
		
	}
	if (threadIdx.x == 0 && blockIdx.x == 0){
		d_tempKinEneVVMax[0] = 0;
		d_tempKinEneVVMax[1] = 0;
	}
}

#define VEL_DOT_PROD_THRDS_PER_BLK 256
const int threadsPerBlock = VEL_DOT_PROD_THRDS_PER_BLK;

__global__ void integratorStep2GPU(ParticlesGpuArgs *args, real dt){

	__shared__ real cacheVVMax[threadsPerBlock];
	__shared__ real cacheVV[threadsPerBlock];
	int cacheIndex = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	VectorR *vel, *acc, *force;
	vel = args->vel;
	acc = args->acc;
	force = args->force;
	real *mass = args->mass;
	real *vv = args->vv;
	real tempMax = 0;
	real tempSum = 0;

	while (i < args->N){
		//Update accelerations
		acc[i].x = force[i].x / mass[i];
		acc[i].y = force[i].y / mass[i];
		//Leap frog step 2
		//VSAdd(v[i], v[i], 0.5*dt, a[i]);
		vel[i].x = vel[i].x + real(0.5)*dt*acc[i].x;
		vel[i].y = vel[i].y + real(0.5)*dt*acc[i].y;

		// calculate v square
		vv[i] = vel[i].x*vel[i].x + vel[i].y*vel[i].y;
		tempMax = Max(tempMax, vv[i]);
		tempSum += mass[i] * vv[i];

		i += blockDim.x * gridDim.x;
	}
	
	cacheVV[cacheIndex] = tempSum;
	cacheVVMax[cacheIndex] = tempMax;
	__syncthreads();
	
	int i1 = blockDim.x / 2;
	while (i1 != 0) {
		if (cacheIndex < i1){
			cacheVV[cacheIndex] += cacheVV[cacheIndex + i1];
			cacheVVMax[cacheIndex] = Max(cacheVVMax[cacheIndex], cacheVVMax[cacheIndex + i1]);
		}
		__syncthreads();
		i1 /= 2;
	}
	if (cacheIndex == 0){
		//vvTemp[blockIdx.x] = cacheVV[0];
		//vvMaxTemp[blockIdx.x] = cacheVVMax[0];
		atomicAdd(&d_tempKinEneVVMax[0], cacheVV[0]);
		atomicMax(&d_tempKinEneVVMax[1], cacheVVMax[0]);
	}
}


__global__ void integratorVelocityVerletStep1GPU(ParticlesGpuArgs *args, VectorR box, real dt, PBCTYPE pbcType){
	// Perform LeapFrog step 1, boundary condition and zero values
	VectorR *pos, *vel, *acc, *force;
	pos = args->pos;
	vel = args->vel;
	acc = args->acc;
	force = args->force;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while (i < args->N){

		pos[i].x = pos[i].x + dt*vel[i].x + real(0.5)*dt*dt*acc[i].x;
		pos[i].y = pos[i].y + dt*vel[i].y + real(0.5)*dt*dt*acc[i].y;

		vel[i].x = vel[i].x + real(0.5)*dt*acc[i].x;
		vel[i].y = vel[i].y + real(0.5)*dt*acc[i].y;

		applyBoundaryCondition(pos[i], box, pbcType);

		//zero stuff
		acc[i].x = 0;
		acc[i].y = 0;
		force[i].x = 0;
		force[i].y = 0;
		args->d_enei[i] = 0;
		i += blockDim.x * gridDim.x;

	}
	if (threadIdx.x == 0 && blockIdx.x == 0){
		d_tempKinEneVVMax[0] = 0;
		d_tempKinEneVVMax[1] = 0;
	}
}


__global__ void integratorVelocityVerletStep2GPU(ParticlesGpuArgs *args, real dt){

	__shared__ real cacheVVMax[threadsPerBlock];
	__shared__ real cacheVV[threadsPerBlock];
	int cacheIndex = threadIdx.x;
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	VectorR *vel, *acc, *force;
	vel = args->vel;
	acc = args->acc;
	force = args->force;
	real *mass = args->mass;
	real *vv = args->vv;
	real tempMax = 0;
	real tempSum = 0;

	while (i < args->N){
		
		//real rx = randonNumber(i, devCurandStates);
		//real ry = randonNumber(i, devCurandStates);
		//real rz = randonNumber(i, devCurandStates);
		//gamma = ? ? ? ;
		//Temperature = ? ? ? ;
		//real coeff = sqrt(6 * gamma*Temperature / dt);
		//force[i].x = rx*coeff - gamma*vel[i].x;
		//force[i].y = ry*coeff - gamma*vel[i].y;
		//force[i].z = rz*coeff - gamma*vel[i].z;

		//Update accelerations
		acc[i].x = force[i].x / mass[i];
		acc[i].y = force[i].y / mass[i];
		//Leap frog step 2
		//VSAdd(v[i], v[i], 0.5*dt, a[i]);
		vel[i].x = vel[i].x + 0.5f*dt*acc[i].x;
		vel[i].y = vel[i].y + 0.5f*dt*acc[i].y;

		// calculate v square
		vv[i] = vel[i].x*vel[i].x + vel[i].y*vel[i].y;
		tempMax = Max(tempMax, vv[i]);
		tempSum += mass[i] * vv[i];

		i += blockDim.x * gridDim.x;
	}

	cacheVV[cacheIndex] = tempSum;
	cacheVVMax[cacheIndex] = tempMax;
	__syncthreads();

	int i1 = blockDim.x / 2;
	while (i1 != 0) {
		if (cacheIndex < i1){
			cacheVV[cacheIndex] += cacheVV[cacheIndex + i1];
			cacheVVMax[cacheIndex] = Max(cacheVVMax[cacheIndex], cacheVVMax[cacheIndex + i1]);
		}
		__syncthreads();
		i1 /= 2;
	}
	if (cacheIndex == 0){
		//vvTemp[blockIdx.x] = cacheVV[0];
		//vvMaxTemp[blockIdx.x] = cacheVVMax[0];
		atomicAdd(&d_tempKinEneVVMax[0], cacheVV[0]);
		atomicMax(&d_tempKinEneVVMax[1], cacheVVMax[0]);
	}
}



void rescaleVelocitiesGPUWrapper(int dimGrid, int dimBlock, VectorR *d_vel_ptr, real factor, int n_particles){
	rescaleVelocitiesGPU << <dimGrid, dimBlock >> >(d_vel_ptr, factor, n_particles);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void zeroGPUWrapper(int dimGrid, int dimBlock, ParticlesGpuArgs *args){
	zeroGPU<<<dimGrid, dimBlock>>>(args);
	HANDLE_ERROR(cudaDeviceSynchronize());
}


void integratorStep1Wrapper(int dimGrid, int dimBlock, ParticlesGpuArgs *args, VectorR box, real dt, PBCTYPE pbcType){
	integratorStep1GPU<<<dimGrid, dimBlock>>>(args, box, dt, pbcType);
	//integratorVelocityVerletStep1GPU << <dimGrid, dimBlock >> >(args, box, dt, pbcType);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

real tempKinEneVVMax[2];
void integratorStep2Wrapper(int dimGrid, int dimBlock, ParticlesGpuArgs *args, real dt){
	integratorStep2GPU << <dimGrid, dimBlock >> >(args, dt);
	//integratorVelocityVerletStep2GPU << <dimGrid, dimBlock >> >(args, dt);
	HANDLE_ERROR(cudaDeviceSynchronize());
	cudaMemcpyFromSymbol(&tempKinEneVVMax, d_tempKinEneVVMax, 2*sizeof(real));
}

__global__ void initRandomGPU(curandState* state, unsigned long seed){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

void initRandomGPUWrapper(int dimGrid, int dimBlock, curandState* devCurandStates){
	
	cudaMalloc(&devCurandStates, dimGrid*dimBlock*sizeof(curandState));
	// setup seeds
	initRandomGPU <<< dimGrid, dimBlock >>> (devCurandStates, time(NULL));
	HANDLE_ERROR(cudaDeviceSynchronize());

}

/*
__global__ void setup_kernel(curandState * state, unsigned long seed)
{
	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void generate(curandState* globalState)
{
	int ind = threadIdx.x;
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
}

int main1(int argc, char** argv)
{
	dim3 tpb(N, 1, 1);
	curandState* devStates;
	cudaMalloc(&devStates, N*sizeof(curandState));

	// setup seeds
	setup_kernel << < 1, tpb >> > (devStates, time(NULL));

	// generate random numbers
	generate << < 1, tpb >> > (devStates);

	return 0;
}

// perform the first half step of velocity verlet
// r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
// v(t+deltaT/2) = v(t) + (1/2)a*deltaT

// a(t+deltaT) gets modified with the bd forces
// v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
{
unsigned int j = m_group->getMemberIndex(group_idx);

// first, calculate the BD forces
// Generate three random numbers
Scalar rx = saru.s<Scalar>(-1,1);
Scalar ry = saru.s<Scalar>(-1,1);
Scalar rz = saru.s<Scalar>(-1,1);

Scalar gamma;
if (m_use_lambda)
gamma = m_lambda*h_diameter.data[j];
else
{
unsigned int type = __scalar_as_int(h_pos.data[j].w);
gamma = h_gamma.data[type];
}

// compute the bd force
Scalar coeff = fast::sqrt(Scalar(6.0) *gamma*currentTemp/m_deltaT);
if (m_noiseless_t)
coeff = Scalar(0.0);
Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

if (D < 3)
bd_fz = Scalar(0.0);

// then, calculate acceleration from the net force
Scalar minv = Scalar(1.0) / h_vel.data[j].w;
h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

// then, update the velocity
h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;


*/

/////////////////////////////////

/* Compute gfric */
//gfric = 1.0 - gamma*dt / 2.0;

/* Compute noise */
//noise = sqrt(6.0*gamma*Tb / dt);

/* First integration half-step 
for (i = 0; i<N; i++) {
	rx[i] += vx[i] * dt + 0.5*dt2*fx[i];
	ry[i] += vy[i] * dt + 0.5*dt2*fy[i];
	rz[i] += vz[i] * dt + 0.5*dt2*fz[i];
	vx[i] = vx[i] * gfric + 0.5*dt*fx[i];
	vy[i] = vy[i] * gfric + 0.5*dt*fy[i];
	vz[i] = vz[i] * gfric + 0.5*dt*fz[i];

	 Apply periodic boundary conditions 
	if (rx[i]<0.0) { rx[i] += L; ix[i]--; }
	if (rx[i]>L)   { rx[i] -= L; ix[i]++; }
	if (ry[i]<0.0) { ry[i] += L; iy[i]--; }
	if (ry[i]>L)   { ry[i] -= L; iy[i]++; }
	if (rz[i]<0.0) { rz[i] += L; iz[i]--; }
	if (rz[i]>L)   { rz[i] -= L; iz[i]++; }
}
 Calculate forces 
 Initialize forces 
for (i = 0; i<N; i++) {
	fx[i] = 2 * noise*(gsl_rng_uniform(r) - 0.5);
	fy[i] = 2 * noise*(gsl_rng_uniform(r) - 0.5);
	fz[i] = 2 * noise*(gsl_rng_uniform(r) - 0.5);
}

PE = total_e(rx, ry, rz, fx, fy, fz, N, L, rc2, ecor, ecut, &vir);

 Second integration half-step 
KE = 0.0;
for (i = 0; i<N; i++) {
	vx[i] = vx[i] * gfric + 0.5*dt*fx[i];
	vy[i] = vy[i] * gfric + 0.5*dt*fy[i];
	vz[i] = vz[i] * gfric + 0.5*dt*fz[i];
	KE += vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
}
*/
/*
Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0 / 2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0 / 2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0 / 2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;

h_pos.data[j].x += dx;
h_pos.data[j].y += dy;
h_pos.data[j].z += dz;
// particles may have been moved slightly outside the box by the above steps, wrap them back into place
box.wrap(h_pos.data[j], h_image.data[j]);

h_vel.data[j].x += Scalar(1.0 / 2.0)*h_accel.data[j].x*m_deltaT;
h_vel.data[j].y += Scalar(1.0 / 2.0)*h_accel.data[j].y*m_deltaT;
h_vel.data[j].z += Scalar(1.0 / 2.0)*h_accel.data[j].z*m_deltaT;

Scalar rx = saru.s<Scalar>(-1, 1);
Scalar ry = saru.s<Scalar>(-1, 1);
Scalar rz = saru.s<Scalar>(-1, 1);

Scalar gamma;
if (m_use_lambda)
gamma = m_lambda*h_diameter.data[j];
else
{
	unsigned int type = __scalar_as_int(h_pos.data[j].w);
	gamma = h_gamma.data[type];
}

// compute the bd force
Scalar coeff = fast::sqrt(Scalar(6.0) *gamma*currentTemp / m_deltaT);
if (m_noiseless_t)
coeff = Scalar(0.0);
Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

if (D < 3)
	bd_fz = Scalar(0.0);

// then, calculate acceleration from the net force
Scalar minv = Scalar(1.0) / h_vel.data[j].w;
h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

// then, update the velocity
h_vel.data[j].x += Scalar(1.0 / 2.0)*h_accel.data[j].x*m_deltaT;
h_vel.data[j].y += Scalar(1.0 / 2.0)*h_accel.data[j].y*m_deltaT;
h_vel.data[j].z += Scalar(1.0 / 2.0)*h_accel.data[j].z*m_deltaT;
*/