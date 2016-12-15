
#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include "particles.h"
#include "md3dsystem.h"


void Particles::init(MDSystem *s){
	sys = s;
	bonds.particles = this;
	feaElements.particles = this;
	N = 0;
	NC = 0;
	numTypes = 1;
	maxTypes = 10;
	bonds.numBonds = 0;
	bonds.bondListLen = 0;
	std::vector<int> all;
	partGroups.push_back(all);
	partGroupLen.push_back(0);
}

void Particles::reset(){
	bonds.reset();
	feaElements.reset();

	pos.clear();
	vel.clear();
	acc.clear();
	force.clear();
	vv.clear();

	radius.clear();
	sigma.clear();
	mass.clear();
	type.clear();
	charge.clear();

	exclusionGroups.clear();
	exclusionGroupLen.clear();
	exclusionGroupIdOfPart.clear();
	int *d_exclusionGroupIdOfPart_ptr;

	clusterIdOfPart.clear();

	ParticlesGpuArgs *h_gpuArgs, *d_gpuArgs;

	//real *h_enei, *d_enei;
	enei.clear();
	viri.clear();

	N = 0;
	numTypes = 1;
	maxTypes = 10;
	bonds.numBonds = 0;
	bonds.bondListLen = 0;
	std::vector<int> all;
	partGroups.clear();
	partGroupLen.clear();
	partGroups.push_back(all);
	partGroupLen.push_back(0);

	if (sys->DEV == GPU){
		
		//d_pos.clear();
		clearDevVector(d_pos);
		//d_vel.clear();
		clearDevVector(d_vel);
		//d_acc.clear();
		clearDevVector(d_acc);
		//d_force.clear();
		clearDevVector(d_force);
		//d_vv.clear();
		clearDevVector(d_vv);

		//d_radius.clear();
		clearDevVector(d_radius);
		//d_sigma.clear();
		clearDevVector(d_sigma);
		//d_mass.clear();
		clearDevVector(d_mass);
		//d_type.clear();
		clearDevVector(d_type);
		//d_charge.clear();
		clearDevVector(d_charge);
		//d_exclusionGroupIdOfPart.clear();
		clearDevVector(d_exclusionGroupIdOfPart);
		cudaFree(d_gpuArgs);
		cudaFree(h_gpuArgs);
		//d_enei.clear();
		clearDevVector(d_enei);
		//d_viri.clear();
		clearDevVector(d_viri);

		
	}
}

void Particles::allocateCuda(){
	if (sys->DEV == GPU){
		d_pos = pos;
		d_vel = vel;
		d_acc = acc;
		d_force = force;
		d_vv = vv;
		d_mass = mass;
		d_type = type;
		d_sigma = sigma;
		d_radius = radius;
		d_exclusionGroupIdOfPart = exclusionGroupIdOfPart;

		d_pos_ptr = thrust::raw_pointer_cast(d_pos.data());
		d_vel_ptr = thrust::raw_pointer_cast(d_vel.data());
		d_acc_ptr = thrust::raw_pointer_cast(d_acc.data());
		d_force_ptr = thrust::raw_pointer_cast(d_force.data());
		d_vv_ptr = thrust::raw_pointer_cast(d_vv.data());
		d_mass_ptr = thrust::raw_pointer_cast(d_mass.data());
		d_type_ptr = thrust::raw_pointer_cast(d_type.data());
		d_sigma_ptr = thrust::raw_pointer_cast(d_sigma.data());
		d_radius_ptr = thrust::raw_pointer_cast(d_radius.data());
		d_exclusionGroupIdOfPart_ptr = thrust::raw_pointer_cast(d_exclusionGroupIdOfPart.data());

		if (bonds.bondList.size() > 0){
			bonds.initBonds();
		} 
		if (feaElements.clusters.size() > 0){
			feaElements.initElements();
		}

		enei.resize(N);
		d_enei = enei;
		d_enei_ptr = thrust::raw_pointer_cast(d_enei.data());

		viri.resize(N);
		d_viri = viri;
		d_viri_ptr = thrust::raw_pointer_cast(d_viri.data());

		cudaMalloc((void **)&(d_gpuArgs), sizeof(ParticlesGpuArgs));
		cudaMallocHost((void **)&(h_gpuArgs), sizeof(ParticlesGpuArgs));

		h_gpuArgs->N = N;
		h_gpuArgs->pos = d_pos_ptr;
		h_gpuArgs->vel = d_vel_ptr;
		h_gpuArgs->acc = d_acc_ptr;
		h_gpuArgs->force = d_force_ptr;
		h_gpuArgs->vv = d_vv_ptr;
		h_gpuArgs->mass = d_mass_ptr;
		h_gpuArgs->type = d_type_ptr;
		h_gpuArgs->sigma = d_sigma_ptr;
		h_gpuArgs->radius = d_radius_ptr;
		h_gpuArgs->exclusionGroupIdOfPart = d_exclusionGroupIdOfPart_ptr;

		h_gpuArgs->bondList = bonds.d_bondList_ptr;
		h_gpuArgs->bondLength = bonds.d_bondLength_ptr;
		h_gpuArgs->numBonds = bonds.numBonds;
		h_gpuArgs->bondListLen = bonds.bondListLen;

		h_gpuArgs->bondListNN = bonds.d_bondListNN_ptr;
		h_gpuArgs->bondLengthNN = bonds.d_bondLengthNN_ptr;
		h_gpuArgs->boundCountNN = bonds.d_bondCountNN_ptr;
		h_gpuArgs->maxBondPartnersNN = bonds.maxBondPartnersNN;

		h_gpuArgs->refPos = feaElements.d_refPos_ptr;
		h_gpuArgs->tetras = feaElements.d_tetras_ptr;
		h_gpuArgs->xm = feaElements.d_xm_ptr;
		h_gpuArgs->ntetras = feaElements.numTetras;
		h_gpuArgs->offset = feaElements.offset;

		h_gpuArgs->d_enei = d_enei_ptr;
		h_gpuArgs->d_viri = d_viri_ptr;

		cudaMemcpy(d_gpuArgs, h_gpuArgs, sizeof(ParticlesGpuArgs), cudaMemcpyHostToDevice);
		printf("Part d_gpuArgs %p\n", d_gpuArgs);
	}
}

void Particles::moveToDevice(){
	// copy all of H back to the beginning of D 
	//thrust::copy(H.begin(), H.end(), D.begin());
	thrust::copy(pos.begin(), pos.end(), d_pos.begin());
	thrust::copy(vel.begin(), vel.end(), d_vel.begin());
	thrust::copy(acc.begin(), acc.end(), d_acc.begin());
	thrust::copy(force.begin(), force.end(), d_force.begin());
	thrust::copy(vv.begin(), vv.end(), d_vv.begin());
}

void Particles::moveToHost(){
	thrust::copy(d_pos.begin(), d_pos.end(), pos.begin());
	//thrust::copy(d_vel.begin(), d_vel.end(), vel.begin());
	//thrust::copy(d_acc.begin(), d_acc.end(), acc.begin());
	//thrust::copy(d_force.begin(), d_force.end(), force.begin());
	thrust::copy(d_vv.begin(), d_vv.end(), vv.begin());
}

int Particles::addParticle(real x, real y, real m, real sig, int typ) {

	int pid = pos.size();
	N += 1;
	sys->N += 1;
	sys->density = sys->N / (sys->box.x*sys->box.y);
	VectorR t;
	t.x = x; t.y = y;
	pos.push_back(t);
	t.x = 0; t.y = 0;
	vel.push_back(t);
	acc.push_back(t);
	force.push_back(t);
	vv.push_back(0);
	mass.push_back(m);
	type.push_back(typ);
	sigma.push_back(sig);
	radius.push_back(sig / 2.0);
	charge.push_back(1);
	insertPartInGroup(pid, 0); //When particles are create, blong to group zero
	exclusionGroupIdOfPart.push_back(-1);
	clusterIdOfPart.push_back(-1);
	return pid;
}

int Particles::addParticle(real x, real y, real vx, real vy, real m, real sig, int typ) {
	int pid = addParticle(x, y, m, sig, typ);
	vel[pid].x = vx;
	vel[pid].y = vy;
	return pid;
}

void Particles::setParticlesVel(real v0){
	for (int p = 0; p < N; p++) {
		//vel[p].x = (RANDOM01 - .5f)*v0;
		//vel[p].y = (RANDOM01 - .5f)*v0;
		real s = RANDOM01;
		s = 2.0*PI*s;
		vel[p].x = cos(s)*v0;
		vel[p].y = sin(s)*v0;

	}
	zeroCM();
}

void Particles::zeroCM(){
	int p;
	VectorR vSum;
	real totMass = 0;;
	vSum.x = vSum.y = 0;
	for (p = 0; p < N; p++){
		vSum.x += vel[p].x * mass[p];
		vSum.y += vel[p].y * mass[p];
		totMass += mass[p];
	}
	// with zero total momentum
	for (p = 0; p < N; p++){
		vel[p].x -= vSum.x / totMass;
		vel[p].y -= vSum.y / totMass;
	}	
}

void Particles::setSigMax(){
	sigMax = sys->particles.sigma[0];
}

int Particles::addGroup() {
	int ind = partGroups.size();
	std::vector<int> g;
	partGroups.push_back(g);
	partGroupLen.push_back(0);
	return ind;
}

void Particles::insertPartInGroup(int pid, int g) {
	groupIdOfPart.push_back(g);
	partGroups[g].push_back(pid);
	partGroupLen[g] += 1;
}

int Particles::addExclusionGroup() {
	int ind = exclusionGroups.size();
	std::vector<int> g;
	exclusionGroups.push_back(g);
	exclusionGroupLen.push_back(0);
	return ind;
}

void Particles::insertPartInExclusionGroup(int pid, int g) {
	exclusionGroupIdOfPart[pid] = g;
	exclusionGroups[g].push_back(pid);
	exclusionGroupLen[g] += 1;
}

void Particles::addElastomer(real *vert, int nvert, int *tet, int ntet, real x, real y, int typ, int g, real sig, real M, bool exclude) {
	int offset = N;
	int pid, clusterInd;
	
	int exclusionGroupId = addExclusionGroup();
	clusterInd = feaElements.addCluster(offset,nvert, ntet);
	feaElements.refPos.resize((clusterInd + 1)*nvert);
	feaElements.unfoldedPos.resize((clusterInd + 1)*nvert);
	//printf("\nAdding elastomer %d\n", e.size);
	for (int i = 0; i < feaElements.clusters[clusterInd].nvertices; i++) {
		real x1 = vert[2*i]   + x;
		real y1 = vert[2*i+1] + y;
		pid = addParticle(x1, y1, 0.0, sig, typ); //FIXME sigma
		feaElements.refPos[i + offset] = pos[i + offset];
		feaElements.unfoldedPos[i + offset] = pos[i + offset];
		applyBoundaryCondition(sys->particles.pos[pid], sys->box, sys->pbcType);
		if (g > 0) insertPartInGroup(pid, g);
		if(exclude) insertPartInExclusionGroup(pid, exclusionGroupId); //FIXME
		clusterIdOfPart[pid] = clusterInd;
	}
	sys->NC += 1;
	NC += 1;

	int nb = 0;
	real totVol = 0;
	int tetsize = ntet;
	int tetOffset = clusterInd*tetsize;
	feaElements.clusters[clusterInd].tetOffset = tetOffset;
	feaElements.tetras.resize((clusterInd + 1)*tetsize);
	feaElements.xm.resize((clusterInd + 1)*tetsize);
	feaElements.refVol.resize((clusterInd + 1)*tetsize);
	for (int i = 0; i < ntet; i++) {
		int i0 = tet[3 * i] + offset - 1;
		int i1 = tet[3 * i + 1] + offset - 1;
		int i2 = tet[3 * i + 2] + offset - 1;

		feaElements.tetras[i + tetOffset][0] = i0;
		feaElements.tetras[i + tetOffset][1] = i1;
		feaElements.tetras[i + tetOffset][2] = i2;

		bonds.createBond(i0, i1);
		bonds.createBond(i0, i2);
		bonds.createBond(i1, i2);
		nb += 3;

		mat3x3 m;
		mat3x3 aa;
		feaElements.xm.push_back(m);
		VectorR r[3];
		for (int j = 0; j < 3; j++) {
			int ip = feaElements.tetras[i + tetOffset].ind[j];
			aa.m[j][0] = 1.0;
			aa.m[j][1] = feaElements.refPos[ip].x;
			aa.m[j][2] = feaElements.refPos[ip].y;
			r[j].x = feaElements.refPos[ip].x;
			r[j].y = feaElements.refPos[ip].y;
		}
		invertMatrix(&aa.m[0][0], &feaElements.xm[i + tetOffset].m[0][0]);
		
		real vol = 0.5*((r[2].y - r[0].y)*(r[1].x - r[0].x) - (r[1].y - r[0].y)*(r[2].x - r[0].x));
		feaElements.refVol[i + tetOffset] = vol;
		totVol += vol;
	}
	feaElements.clusters[clusterInd].mass = M;
	feaElements.clusters[clusterInd].rho = M / totVol;
	feaElements.numTetras = feaElements.tetras.size();
	for (int itet = 0; itet < tetsize; itet++){
		for (int j = 0; j < 3; j++) {
			int ip = feaElements.tetras[itet+tetOffset][j];
			mass[ip] += feaElements.clusters[clusterInd].rho*feaElements.refVol[itet] / 3.0;
		}
	}
	
	
	//real vol0 = 0;
	//real newVol = e.refPos.size()*PI*0.8;
	//real rad = sqrt(0.8*e.totVol / (e.refPos.size()*PI));
	//for (int itet = 0; itet < e.tetras.size(); itet++) {
	//	for (int j = 0; j < 3; j++) {
	//		int ip = e.tetras[itet][j]+offset;
	//		mass[ip] += e.erho*e.tetVol[itet]/3.0;
			//sigma[ip] = 2*rad;
			//radius[ip] = sigma[ip]/2;
	//	}
		//printf("%f\n", e.tetVol[itet]);
		//vol0 += e.tetVol[itet];
	//}
	//e.id = elastomers.size();
	//elastomers.push_back(e);
	//printf("Elastomer added vertices: %d tetras: %d bonds: %d %f %f\n", e.refPos.size(), e.tetras.size(), nb, e.totVol, sigma[0]);
}

void Particles::checkOverlap(){
	for (int i = 0; i<sys->N; i++){
		for (int j = 0; j<sys->N; j++){
			//console.log(this.N,i,j);
			//int gi = sys->particles.exclusionGroupIdOfPart[i];
			//int gj = sys->particles.exclusionGroupIdOfPart[j];
			//if ((i != j) && (gi == -1 || gj == -1 || (gi != gj))){
			if (i != j){
				real sigmaij = (sigma[i] + sigma[j]) / 2;

				Box L = sys->box;
				real dx = pos[i].x - pos[j].x;
				real dy = pos[i].y - pos[j].y;
				//if (sys->pbc) {
				if (0){
					if (dx > 0.5*L.x) dx -= L.x;
					else if (dx < -0.5*L.x) dx += L.x;
					if (dy > 0.5*L.y) dy -= L.y;
					else if (dy < -0.5*dy) dy += L.y;
				};
				real dr2 = dx*dx + dy*dy;
				real rCut2 = (0.95*sigmaij)*(0.95*sigmaij);
				if (dr2 < rCut2){
					printf("overlap: %d %d %f\n", i, j, dr2);
				}
			}
		}
	}
}

