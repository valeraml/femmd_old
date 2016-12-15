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

#include "bonds.h"

void Bonds::createBond(int atomi, int atomj) {
	bondList.push_back(atomi);
	bondList.push_back(atomj);
	real dx = particles->pos[atomi].x - particles->pos[atomj].x;
	real dy = particles->pos[atomi].y - particles->pos[atomj].y;
	real r = sqrt(dx*dx + dy*dy);
	bondLength.push_back(r);
	//if (particles->radius[atomi] < r / 2) { particles->radius[atomi] = r / 2; particles->sigma[atomi] = 2 * particles->radius[atomi]; }
	//if (particles->radius[atomj] < r / 2) { particles->radius[atomj] = r / 2; particles->sigma[atomj] = 2 * particles->radius[atomj]; }
	//printf("bond: %d %d %f\n", atomi, atomj, r);
	//particles->radius[atomi] = 0.95*r / 2; particles->sigma[atomi] = 2 * particles->radius[atomi]; 
	//particles->radius[atomj] = 0.95*r / 2; particles->sigma[atomj] = 2 * particles->radius[atomj]; 
	numBonds += 1;
	bondListLen += 2;
	//printf("%f (%d, %d) %d %d\n", r, atomi, atomj, numBonds, bondListLen);
}

void Bonds::initBonds(){
	int N = particles->N;
	maxBondPartnersNN = 300;

	dimGrid = (numBonds + 32) / 32;
	dimBlock = 32;
	d_bondList = bondList;
	d_bondLength = bondLength;
	d_bondList_ptr = thrust::raw_pointer_cast(d_bondList.data());
	d_bondLength_ptr = thrust::raw_pointer_cast(d_bondLength.data());

	std::vector<int> tempBondList(N*maxBondPartnersNN, 0);
	std::vector<real> tempBondLengthList(N*maxBondPartnersNN, 0);
	std::vector<int> tempBondCount(N, 0);

	for (int b = 0; b < numBonds; b++){
		int i = bondList[2 * b];
		int j = bondList[2 * b + 1];

		int jInd = tempBondCount[i];
		tempBondList[i*maxBondPartnersNN + jInd] = j;
		tempBondLengthList[i*maxBondPartnersNN + jInd] = bondLength[b];
		tempBondCount[i] += 1;
		//printf("x %d (%d %d) %d\n", b, i, j, tempBondCount[i]);
		int iInd = tempBondCount[j];
		tempBondList[j*maxBondPartnersNN + iInd] = i;
		tempBondLengthList[j*maxBondPartnersNN + iInd] = bondLength[b];
		tempBondCount[j] += 1;
		//printf("xx %d (%d %d) %f (%d %d)\n\n", b,i,j,bondLength[b],tempBondCount[i],tempBondCount[j]);
	}
	int oldMaxBondPartners = maxBondPartnersNN;
	maxBondPartnersNN = *max_element(tempBondCount.begin(), tempBondCount.end());
	int sum = std::accumulate(tempBondCount.begin(), tempBondCount.end(), 0);
	printf("total bonds: %d\n", sum);
	bondListNN.resize(N*maxBondPartnersNN);
	bondLengthNN.resize(N*maxBondPartnersNN);
	bondCountNN.resize(N);
	for (int i = 0; i < N; i++){
		bondCountNN[i] = tempBondCount[i];
		for (int j = 0; j < bondCountNN[i]; j++){
			bondListNN[i*maxBondPartnersNN + j] = tempBondList[i*oldMaxBondPartners + j];
			bondLengthNN[i*maxBondPartnersNN + j] = tempBondLengthList[i*oldMaxBondPartners + j];
			//printf("(%d %d) %d->%d %f\n",i,j,i,bondListNN[i*maxBondPartnersNN+j],bondLengthNN[i*maxBondPartnersNN+j]);
		}
	}
	d_bondListNN = bondListNN;
	d_bondLengthNN = bondLengthNN;
	d_bondCountNN = bondCountNN;
	d_bondListNN_ptr = thrust::raw_pointer_cast(d_bondListNN.data());
	d_bondLengthNN_ptr = thrust::raw_pointer_cast(d_bondLengthNN.data());
	d_bondCountNN_ptr = thrust::raw_pointer_cast(d_bondCountNN.data());

}


void Bonds::reset(){
	bondList.clear();
	bondLength.clear();
	clearDevVector(d_bondList);
	clearDevVector(d_bondLength);

	bondListNN.clear();
	bondLengthNN.clear();
	bondCountNN.clear();
	clearDevVector(d_bondListNN);
	clearDevVector(d_bondLengthNN);
	clearDevVector(d_bondCountNN);
	numBonds = 0;
	bondListLen = 0;
}