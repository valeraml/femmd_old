#ifndef __BONDS_H__
#define __BONDS_H__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_types.h"
#include "def.h"

class Particles;

class Bonds{

public:
	Particles *particles;
	int dimGrid, dimBlock;
	int numBonds;
	int bondListLen;
	hvector<int> bondList;
	hvector<real> bondLength;
	dvector<int> d_bondList;
	dvector<real> d_bondLength;
	int *d_bondList_ptr;
	real *d_bondLength_ptr;

	hvector<int>bondListNN;
	hvector<real>bondLengthNN;
	hvector<int>bondCountNN;
	dvector<int>d_bondListNN;
	dvector<real>d_bondLengthNN;
	dvector<int>d_bondCountNN;
	int *d_bondListNN_ptr, *d_bondCountNN_ptr;
	real *d_bondLengthNN_ptr;
	int maxBondPartnersNN;

	void createBond(int atomi, int atomj);
	void initBonds();
	void reset();

};


#endif