#ifndef __PARTICLES_H__
#define __PARTICLES_H__


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_types.h"
#include "def.h"
#include "bonds.h"
#include "feaElements.h"

class MDSystem;

struct ParticlesGpuArgs{
	int N;
	int numTypes;
	VectorR *pos, *vel, *acc, *force;
	real *sigma, *mass, *vv, *radius;
	int *type, *exclusionGroupIdOfPart;	

	int *bondList;
	real *bondLength;
	int numBonds;
	int bondListLen;

	int *bondListNN;
	int *boundCountNN;
	real *bondLengthNN;
	int maxBondPartnersNN;

	VectorR *refPos;
	tetraIndexes *tetras;
	mat3x3 *xm;
	int ntetras;
	int offset;

	real *d_enei;
	real *d_viri;
};

class Particles{
public:
	MDSystem *sys;
	int N;
	int NC;
	int bDimBlock, bDimGrid;

	hvector<VectorR> pos;
	hvector<VectorR> vel;
	hvector<VectorR> acc;
	hvector<VectorR> force;
	hvector<real> vv;

	hvector<real> radius;
	hvector<real> sigma;
	hvector<real> mass;
	hvector<real> charge;
	hvector<int> type;

	dvector<VectorR> d_pos;
	dvector<VectorR> d_vel;
	dvector<VectorR> d_acc;
	dvector<VectorR> d_force;
	dvector<real> d_vv;

	dvector<real> d_radius;
	dvector<real> d_sigma;
	dvector<real> d_mass;
	dvector<real> d_charge;
	dvector<int> d_type;

	VectorR *d_pos_ptr, *d_vel_ptr, *d_acc_ptr, *d_force_ptr;
	real *d_mass_ptr, *d_sigma_ptr, *d_vv_ptr, *d_radius_ptr;
	int *d_type_ptr;

	Bonds bonds;
	FeaElements feaElements;

	//std::vector<Elastomer> elastomers;
	//std::vector<int> elastomerId;

	int numTypes;
	int maxTypes;
	real sigMax;

	std::vector< std::vector<int> > partGroups;
	std::vector<int> partGroupLen;
	std::vector<int> groupIdOfPart; //array with ids of group that particle belong to

	std::vector< std::vector<int> > exclusionGroups;
	std::vector<int> exclusionGroupLen;
	hvector<int> exclusionGroupIdOfPart;
	dvector<int> d_exclusionGroupIdOfPart;
	int *d_exclusionGroupIdOfPart_ptr;

	hvector<int> clusterIdOfPart;

	ParticlesGpuArgs *h_gpuArgs, *d_gpuArgs;

	//real *h_enei, *d_enei;
	hvector<real> enei;
	dvector<real> d_enei;
	real *h_enei_ptr, *d_enei_ptr;

	hvector<real> viri;
	dvector<real> d_viri;
	real *h_viri_ptr, *d_viri_ptr;

	void init(MDSystem *sys);
	void reset();
	void allocateCuda();
	void moveToDevice();
	void moveToHost();

	int addParticle(real x, real y, real m, real sig, int typ);
	int addParticle(real x, real y, real vx, real vy, real m, real sig, int typ);
	void setParticlesVel(real v0);
	void zeroCM();
	int addGroup();
	void insertPartInGroup(int pid, int g);
	int addExclusionGroup();
	void insertPartInExclusionGroup(int pid, int g);
	void addElastomer(real *vert, int nvert, int *tet, int ntet, real x, real y, int typ, int g, real sig, real m, bool exclude);
	void setSigMax();
	void checkOverlap();
};

#endif
