
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "nn.h"
#include "md3dsystem.h"

void updateCellsCPU(hvector<VectorR> &r, VectorR &box, VectorI &numCells2D, int n_particles, int maxParticlesPerCell, 
				hvector<int> &cellCount, hvector<int> &cellContent){
	int cId, particlesInCellcId;
	//#pragma omp parallel for
	for (int pi = 0; pi< n_particles; pi++){
		cId = cellId(r[pi], box, numCells2D);
		particlesInCellcId = cellCount[cId];
		cellCount[cId]++;
		int ind = cId*maxParticlesPerCell + particlesInCellcId;
		//#pragma omp critical
		cellContent[ind] = pi;
	}
}

void buildNNCPUSerial(hvector<VectorR> &r, VectorR &box, PBCTYPE pbcType, hvector<int> &cellCount, hvector<int> &cellContent,
	VectorI &numCells2D, int maxParticlesPerCell, int n_particles, real nnRadiusSq,
	hvector<int> &neighborList, int maxNeighborsPerParticle, hvector<int> &neighborsCount, hvector<int> &exclusionGroupIdOfPart){
	//#pragma omp parallel for
	for (int pi = 0; pi<n_particles; pi++){
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
					int gi = exclusionGroupIdOfPart[pi];
					int gj = exclusionGroupIdOfPart[pj];
					if ((pi != pj) && (gi == -1 || gj == -1 || (gi != gj))){
					//if (pi != pj){
						dr.x = r[pi].x - r[pj].x;
						dr.y = r[pi].y - r[pj].y;
						nearestImage(dr, box, pbcType);
						dr2 = dr.x*dr.x + dr.y*dr.y;
						if (dr2 <= nnRadiusSq){
							neighborList[pi*maxNeighborsPerParticle + neighbors] = pj;
							neighbors++;
						}
					}
				}
			}
		}
		neighborsCount[pi] = neighbors;
	}
}



void buildNNCPUSerialAll(hvector<VectorR> &r, VectorR &box, PBCTYPE pbcType, hvector<int> &cellCount, hvector<int> &cellContent,
	VectorI &numCells3D, int maxParticlesPerCell, int n_particles, real nnRadiusSq,
	hvector<int> &neighborList, int maxNeighborsPerParticle, hvector<int> &neighborsCount, hvector<int> &exclusionGroupIdOfPart){

	VectorR dr;
	real dr2;
	int neighbors;
	for (int pi = 0; pi<n_particles; pi++){
		neighbors = 0;
		for (int pj = 0; pj<n_particles; pj++){
			int gi = exclusionGroupIdOfPart[pi];
			int gj = exclusionGroupIdOfPart[pj];
			if ((pi != pj) && (gi == -1 || gj == -1 || (gi != gj))){
			//if (pi != pj){
				dr.x = r[pi].x - r[pj].x;
				dr.y = r[pi].y - r[pj].y;
				nearestImage(dr, box, pbcType);
				dr2 = dr.x*dr.x + dr.y*dr.y;
				if (dr2 < nnRadiusSq){
					neighborList[pi*maxNeighborsPerParticle + neighbors] = pj;
					neighbors++;
				}
			}
		}
		neighborsCount[pi] = neighbors;
	}
}


void NeighborList::buildNNCPU(){
	
	memset(&cellCount[0], 0, cellCount.size() * sizeof(cellCount[0]));
	memset(&cellContent[0], 0, cellContent.size() * sizeof(cellCount[0]));
	memset(&neighborList[0], 0, neighborList.size() * sizeof(neighborList[0]));
	memset(&neighborCount[0], 0, neighborCount.size() * sizeof(neighborCount[0]));

	updateCellsCPU(sys->particles.pos, sys->box, numCells2D, sys->N,
		maxParticlesPerCell, cellCount, cellContent);

	buildNNCPUSerial(sys->particles.pos, sys->box, sys->pbcType, cellCount, cellContent, numCells2D, maxParticlesPerCell,
		sys->N, nnRadiusSq, neighborList, maxNeighborsPerParticle, neighborCount, sys->particles.exclusionGroupIdOfPart);

	//buildNNCPUSerialAll(sys->particles.pos, sys->box, sys->pbcType, cellCount, cellContent, numCells3D, maxParticlesPerCell,
	//				sys->N, nnRadiusSq, neighborList, maxNeighborsPerParticle, neighborCount, sys->particles.exclusionGroupIdOfPart);

}


void NeighborList::update(int restart){
	//printf("xxx list %d %d %f %f\n", totalUpdates, totalSteps, dispHi, 0.5*skin);
	//buildNN();
	if (restart)initialUpdate = 1;
	if (dispHi > 0.5*skin) {
		totalUpdates++;
		totalSteps = sys->steps - initialSteps;
		//printf("updating list %d %d %d %f %f\n", sys->steps, totalUpdates, totalSteps, dispHi, (real) totalSteps / totalUpdates);
		dispHi = 0.0;
		buildNN();
	}
	if (initialUpdate){
		totalUpdates = 1;
		totalSteps = 0;
		initialSteps = sys->steps;
		initialUpdate = 0;
		dispHi = 0;
		buildNN();
		//printf("initial update list at %d\n",sys->steps);
	}
}


void NeighborList::buildNN(){
	if (sys->DEV == GPU){
		buildNNGPU();
	}else{
		buildNNCPU();
	}
}


void NeighborList::allocate(){
	//Cup allocaction
	//h_cellCount = (int *)malloc(numCells*sizeof(int));
	//h_cellContent = (int *)malloc(maxParticlesPerCell*numCells*sizeof(int));
	//h_neighborList = (int *)malloc(sys->n_particles*maxNeighborsPerParticle*sizeof(int));
	//h_neighborCount = (int *)malloc(sys->n_particles*sizeof(int));

	cellCount.resize(numCells);
	cellContent.resize(maxParticlesPerCell*numCells);
	neighborList.resize(sys->N*maxNeighborsPerParticle);
	neighborCount.resize(sys->N);

	memset(&cellCount[0], 0, cellCount.size() * sizeof(cellCount[0]));
	memset(&cellContent[0], 0, cellContent.size() * sizeof(cellCount[0]));
	memset(&neighborList[0], 0, neighborList.size() * sizeof(neighborList[0]));
	memset(&neighborCount[0], 0, neighborCount.size() * sizeof(neighborCount[0]));

	//cudaAllocate();
	if (sys->DEV == GPU){
		d_cellContent = cellContent;
		d_cellCount = cellCount;
		d_neighborList = neighborList;
		d_neighborCount = neighborCount;
		
		d_cellContent_ptr = thrust::raw_pointer_cast(d_cellContent.data());
		d_cellCount_ptr = thrust::raw_pointer_cast(d_cellCount.data());
		d_neighborList_ptr = thrust::raw_pointer_cast(d_neighborList.data());
		d_neighborCount_ptr = thrust::raw_pointer_cast(d_neighborCount.data());

		printf("%p %p %p %p\n", d_cellCount_ptr, d_cellContent_ptr, d_neighborList_ptr, d_neighborCount_ptr);
		cudaMalloc((void **)&(d_nnGpuArgs), sizeof(NNGpuArgs));
		cudaMallocHost((void **)&(h_nnGpuArgs), sizeof(NNGpuArgs));
		h_nnGpuArgs->pCellContent = d_cellContent_ptr;
		h_nnGpuArgs->pCellCount = d_cellCount_ptr;
		h_nnGpuArgs->pNeighborList = d_neighborList_ptr;
		h_nnGpuArgs->pNeighborCount = d_neighborCount_ptr;
		h_nnGpuArgs->maxParticlesPerCell = maxParticlesPerCell;
		h_nnGpuArgs->maxNeighborsPerParticle = maxNeighborsPerParticle;
		h_nnGpuArgs->nnRadiusSq = nnRadiusSq;
		h_nnGpuArgs->numCells = numCells;
		h_nnGpuArgs->numCells2D.x = numCells2D.x;
		h_nnGpuArgs->numCells2D.y = numCells2D.y;
		cudaMemcpy(d_nnGpuArgs, h_nnGpuArgs, sizeof(NNGpuArgs), cudaMemcpyHostToDevice);
	}
}

void NeighborList::init(){
	//normalize to max sigma;
	//real s = sys->particles.sigMax;
	//printf("sigmaMax %f\n", s);
	real s = 1.0;
	init(s*sys->neighborlist.skin, s*sys->interactions.rCutMax);
}

void NeighborList::init(real s, real rc){
	//dimGrid = sys->dimGrid;
	//dimBlock = sys->dimBlock;
	N = sys->n_particles = sys->N;
	skin = s;
	rCut = rc; //FIXME incorporate sigma
	rCutSq = rc*rc;
	nnRadius = (rc + s);
	dispHi = 0;
	totalUpdates = 0;
	initialUpdate = 1;
	nnRadiusSq = nnRadius*nnRadius;
	numCells2D.x = int(floor(sys->box.x / nnRadius));
	numCells2D.y = int(floor(sys->box.y / nnRadius));
	real vCell = (sys->box.x / numCells2D.x)*(sys->box.y / numCells2D.y);
	real factor = 2;
	real dens = sys->density;
	if (dens < 1.0) dens = 1.0;
	numCells = numCells2D.x*numCells2D.y;
	real temp1 = (factor*dens*vCell);
	maxParticlesPerCell = temp1;
	//(approx 4/3*pi ~ 5)
	//maxNeighborsPerParticle = (int)(factor * 5.0*nnRadius*nnRadius*nnRadius*dens);
	real temp = (factor * 5.0*nnRadius*nnRadius*nnRadius*dens);
	maxNeighborsPerParticle = temp;
	printf("Neighbor List Initialized: %d %d %d\n", maxParticlesPerCell, maxNeighborsPerParticle, numCells);

	allocate();

	///Print memory usage/////
	/*
	real memPos = sys->n_particles*sizeof(VectorR);
	real memCellCount = numCells*sizeof(int);
	real memCellContent = maxParticlesPerCell*numCells*sizeof(int);
	real memNN = sys->n_particles*maxNeighborsPerParticle*sizeof(int);
	real memCount = sys->n_particles*sizeof(int);
	real MB = 1000 * 1000;
	real GB = 1000 * MB;

	printf("========== memory usage by nn =====================\n");
	printf("Num Part: %d, Num Cells %d, (%d %d), Max PPC %d\n", sys->n_particles, numCells,
		numCells2D.x, numCells2D.y, maxParticlesPerCell);
	printf("memroy usage by particles:          %f Bytes, %f MB %f GB\n", 4 * memPos, 4 * memPos / MB, 4 * memPos / GB);
	printf("memroy usage by NN CellCount:       %f Bytes, %f MB %f GB\n", memCellCount, memCellCount / MB, memCellCount / GB);
	printf("memroy usage by NN memCellContent : %f Bytes, %f MB %f GB\n", memCellContent, memCellContent / MB, memCellContent / GB);
	printf("memroy usage by NN memPairs:        %f Bytes, %f MB %f GB\n", memNN, memNN / MB, memNN / GB);
	printf("memroy usage by NN memVPairs:       %f Bytes, %f MB %f GB\n", memCount, memCount / MB, memCount / GB);
	printf("Number of threads dimGrid: %d dimBlock: %d threads: %d\n", sys->dimGrid, sys->dimBlock, sys->dimGrid*sys->dimBlock);
	*/

};

void NeighborList::reset(){
	cellCount.clear();
	cellContent.clear();
	neighborList.clear();
	neighborCount.clear();
	if (sys->DEV == GPU){
		//d_cellContent.clear();
		clearDevVector(d_cellContent);
		//d_cellCount.clear();
		clearDevVector(d_cellCount);
		//d_neighborList.clear();
		clearDevVector(d_neighborList);
		//d_neighborCount.clear();
		clearDevVector(d_neighborCount);
		cudaFree(d_nnGpuArgs);
		cudaFree(h_nnGpuArgs);
	}

}
