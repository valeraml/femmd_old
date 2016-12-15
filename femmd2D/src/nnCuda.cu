
#include "cuda_runtime.h"
#include "nn.h"
#include "md3dsystem.h"
#include "CudaUtil.h"



__device__ __host__ int cellId(VectorR &r, VectorR &box, VectorI &numCells2D){
	int ix, iy, iz, cellId;
	//FIXME the ifs are a hack, if x = -0, then bc sends then to box size and cell is out of range
	ix = r.x*numCells2D.x / box.x; if (ix == numCells2D.x) ix -= 1;
	iy = r.y*numCells2D.y / box.y; if (iy == numCells2D.y) iy -= 1;
	cellId = ix + iy*numCells2D.x;
	//printf("for r = (%f %f %f) get (%d %d %d) %d\n",r.x,r.y,r.z,ix,iy,iz,cellId);
	return cellId;
}

__device__ __host__ void cell3DIndex(int cellId, VectorI &cell2DId, VectorI &numCells2D){
	cell2DId.y = cellId / numCells2D.x;
	cellId -= cell2DId.y*numCells2D.x;
	cell2DId.x = cellId;
	//printf("for cellId %d (%d %d %d) \n",ti, cell3DId->x, cell3DId->y, cell3DId->z);
}

__device__ __host__ int applyCellBC(VectorI &curCell, VectorI &numCells2D, PBCTYPE pbcType){
	
	//if (pbcType == XYZPBC){
		if (curCell.x < 0) curCell.x = numCells2D.x - 1;
		else if (curCell.x == numCells2D.x) curCell.x = 0;

		if (curCell.y < 0) curCell.y = numCells2D.y - 1;
		else if (curCell.y == numCells2D.y) curCell.y = 0;

	//}
	return 0;
	
	/*
	int re = 0;
	switch (pbcType){
	case XYZPBC:
		if (curCell.x < 0) curCell.x = numCells3D.x - 1;
		else if (curCell.x == numCells3D.x) curCell.x = 0;
		if (curCell.y < 0) curCell.y = numCells3D.y - 1;
		else if (curCell.y == numCells3D.y) curCell.y = 0;
		if (curCell.z < 0) curCell.z = numCells3D.z - 1;
		else if (curCell.z == numCells3D.z) curCell.z = 0;
		break;
	case XYPBC:
		if (curCell.x < 0) curCell.x = numCells3D.x - 1;
		else if (curCell.x == numCells3D.x) curCell.x = 0;
		if (curCell.y < 0) curCell.y = numCells3D.y - 1;
		else if (curCell.y == numCells3D.y) curCell.y = 0;
		if (curCell.z < 0 ||curCell.z == numCells3D.z) re = 1;
		break;
	case NOPBC:
		if (curCell.x < 0 || curCell.z == numCells3D.x) re = 1;
		if (curCell.y < 0 || curCell.z == numCells3D.y) re = 1;
		if (curCell.z < 0 || curCell.z == numCells3D.z) re = 1;
		re = 1;
		break;
	default:
		break;
	}
	return re;
	*/
}

__global__ void updateCellsGPU(VectorR *r, VectorR box, VectorI numCells2D, int n_particles, int maxParticlesPerCell, int *cellCount, int *cellContent){
//__global__ void updateCellsGPU(NNGpuArgs *nnArgs, VectorR *r, VectorR &box, int n_particles){
//	int *cellCount = nnArgs->pCellCount;
//	int *cellContent = nnArgs->pCellContent;
//	VectorI numCells3D = nnArgs->numCells3D;
//	int maxParticlesPerCell = nnArgs->maxParticlesPerCell;

	int pi = threadIdx.x + blockDim.x*blockIdx.x;
	if (pi < n_particles){
		int cId = cellId(r[pi], box, numCells2D);
		int particlesInCellcId = atomicAdd(&cellCount[cId], 1);
		cellContent[cId*maxParticlesPerCell + particlesInCellcId] = pi;
	}
}

__global__ void buildNNGPUKernel(ParticlesGpuArgs *pArgs, NNGpuArgs *d_nnGpuArgs, VectorR box, PBCTYPE pbcType, int n_particles){

	int *cellCount = d_nnGpuArgs->pCellCount;
	int *cellContent = d_nnGpuArgs->pCellContent;
	VectorI numCells2D = d_nnGpuArgs->numCells2D;
	int maxParticlesPerCell = d_nnGpuArgs->maxParticlesPerCell;
	real nnRadiusSq = d_nnGpuArgs->nnRadiusSq;
	int *d_neighborList = d_nnGpuArgs->pNeighborList;
	int maxNeighborsPerParticle = d_nnGpuArgs->maxNeighborsPerParticle;
	int *d_neighborsCount = d_nnGpuArgs->pNeighborCount;
	VectorR *r = pArgs->pos;

	int pi = threadIdx.x + blockDim.x*blockIdx.x;
	if (pi < n_particles){
		int neighborCellId, npCell, pj;
		VectorR dr;
		real dr2;
		VectorI piCell;
		VectorI neighborCell;
		int piCellId = cellId(r[pi], box, numCells2D);
		cell3DIndex(piCellId, piCell, numCells2D);

		int neighbors = 0;
		for (int cx = -1; cx <= 1; cx++){
			for (int cy = -1; cy <= 1; cy++){
					neighborCell.x = piCell.x + cx;
					neighborCell.y = piCell.y + cy;
					if(applyCellBC(neighborCell, numCells2D, pbcType))continue;
					neighborCellId = neighborCell.x + neighborCell.y*numCells2D.x;
					npCell = cellCount[neighborCellId];
					for (int j = 0; j<npCell; j++){
						pj = cellContent[neighborCellId*maxParticlesPerCell + j];
						int gi = pArgs->exclusionGroupIdOfPart[pi];
						int gj = pArgs->exclusionGroupIdOfPart[pj];
						if ((pi != pj) && (gi == -1 || gj == -1 || (gi != gj))){
						//if (pi != pj){
							dr.x = r[pi].x - r[pj].x;
							dr.y = r[pi].y - r[pj].y;
							nearestImage(dr, box, pbcType);
							dr2 = dr.x*dr.x + dr.y*dr.y;
							if (dr2 <= nnRadiusSq){
								d_neighborList[pi*maxNeighborsPerParticle + neighbors] = pj;
								neighbors++;
							}
						}
					}
			}
		}
		d_neighborsCount[pi] = neighbors;
	}

}

void NeighborList::buildNNGPU(){

	HANDLE_ERROR(cudaMemset(d_cellCount_ptr, 0, numCells*sizeof(int)));
	//HANDLE_ERROR(cudaMemset(d_cellContent_ptr, 0, maxParticlesPerCell*numCells*sizeof(int)));
	//HANDLE_ERROR(cudaMemset(d_neighborList_ptr, 0, sys->N*maxNeighborsPerParticle*sizeof(int)));
	HANDLE_ERROR(cudaMemset(d_neighborCount_ptr, 0, sys->N*sizeof(int)));
	HANDLE_ERROR(cudaDeviceSynchronize());

	updateCellsGPU <<< sys->dimGrid, sys->dimBlock >>>(sys->particles.d_pos_ptr, sys->box, numCells2D, sys->n_particles,
		maxParticlesPerCell, d_cellCount_ptr, d_cellContent_ptr);
	//updateCellsGPU <<< sys->dimGrid, sys->dimBlock >>>(d_nnGpuArgs, sys->particles.d_pos_ptr, sys->box, sys->N);
	HANDLE_ERROR(cudaDeviceSynchronize());
	buildNNGPUKernel <<<sys->dimGrid, sys->dimBlock >>>(sys->particles.d_gpuArgs, d_nnGpuArgs, sys->box, sys->pbcType, sys->N );
	HANDLE_ERROR(cudaDeviceSynchronize());

};

