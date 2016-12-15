#ifndef __NEIGHBORLIST_H__
#define __NEIGHBORLIST_H__

#include "def.h"

class MDSystem;

struct NNGpuArgs{
	int *pNeighborList;
	int *pNeighborCount;
	int *pCellCount;
	int *pCellContent;
	int maxParticlesPerCell;
	int maxNeighborsPerParticle;
	real nnRadiusSq;
	VectorI numCells2D;
	int numCells;
};


class NeighborList{

public:
	int N; //FIXME take this out of here
	MDSystem *sys;
	int totalUpdates;
	int initialSteps;
	int totalSteps;
	int initialUpdate;
	real dispHi;
	real skin;
	real rCut;
	real rCutSq;
	real nnRadius;
	real nnRadiusSq;

	int numCells;
	VectorI numCells2D;

	int maxParticlesPerCell;
	hvector<int> cellCount;
	hvector<int> cellContent;
	dvector<int> d_cellCount;
	dvector<int> d_cellContent;

	int maxNeighborsPerParticle;
	hvector<int> neighborList;
	hvector<int> neighborCount;
	dvector<int> d_neighborList;
	dvector<int> d_neighborCount;

	int *d_neighborList_ptr; 
	int *d_neighborCount_ptr; 
	int *d_cellCount_ptr; 
	int *d_cellContent_ptr;

	NNGpuArgs *h_nnGpuArgs, *d_nnGpuArgs;

	//Radix sort test
	//CellList *cellList;
	//int *cellListCount;

	NeighborList(){ dispHi = 0; rCut = 0.0; skin = 0.4; totalUpdates = 0; initialUpdate = 1; }
	void setSkin(real s){ skin = s; };
	void init(MDSystem *s){ sys = s; }
	void init(real s, real rc);
	void init();
	void reset();
	void allocate();
	//void cudaAllocate();
	void update(int restart);
	//void buildNNGPU();
	void buildNNCPU();
	void buildNNGPU();
	void buildNN();

};

//void nearestImage(VectorR &dr, VectorR &box, PBCTYPE pbcType);
int cellId(VectorR &r, VectorR &box, VectorI &numCells2D);
void cell3DIndex(int cellId, VectorI &cell2DId, VectorI &numCells2D);
int applyCellBC(VectorI &curCell, VectorI &numCells2D, PBCTYPE pbcType);



#endif

