#ifndef __FEAELEMENT_H__
#define __FEAELEMENT_H__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_types.h"
#include "def.h"

class Cluster{
public:
	int offset;
	int tetOffset;
	int nvertices;
	int ntetras;
	real currVol;
	real refVo;
	real rho;
	real mass;
	VectorR cmPos;
	VectorR cmVel;
	VectorR centroid;
};

class FeaElements{
public:

	Particles *particles;
	int dimGrid, dimBlock;
	int numTetras;
	int offset;

	hvector<Cluster> clusters;
	
	hvector< VectorR > refPos;
	hvector< VectorR > unfoldedPos;
	hvector< tetraIndexes > tetras;
	hvector< mat3x3 > xm;
	hvector< real > refVol;
	hvector< real > currVol;

	dvector< VectorR> d_refPos;
	dvector< tetraIndexes > d_tetras;
	dvector< mat3x3 > d_xm;
	dvector< real > d_refVol;

	VectorR *d_refPos_ptr;
	tetraIndexes *d_tetras_ptr;
	mat3x3* d_xm_ptr;
	real* d_refVol_ptr;

	int tetraListLen;
	hvector<tetraIndexes> tetraNN;
	dvector<tetraIndexes> d_tetraNN;
	tetraIndexes* d_tetraNN_ptr;

	void initElements();
	void reset();
	void unfoldPos(VectorR box);
	void checkAreas(VectorR box);
	void updateClustersCentroid(VectorR &box);
	void calcClusterProps();

	int addCluster(int offset, int nv, int nt){
		int ind = clusters.size();
		clusters.resize(clusters.size() + 1);
		clusters[ind].offset = offset;
		clusters[ind].nvertices = nv;
		clusters[ind].ntetras = nt;
		return ind;
	}

};


#endif