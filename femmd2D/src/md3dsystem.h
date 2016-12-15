#ifndef __MD3DSYSTEM_H__
#define __MD3DSYSTEM_H__

#include <vector>
#include "def.h"
#include "particles.h"
#include "nn.h"
#include "interactions.h"
#include "integrator.h"
#include "properties.h"

// Properties

class MDSystem {

public:
	device DEV;
	bool start;
	int dimGrid, dimBlock;
	int N;
	int NC;
	int n_particles;

	Particles particles;
	NeighborList neighborlist;
	Interactions interactions;
	Integrator integrator;
	Properties props;

	int steps;
	real dt;
	real simTime;
	int thermalizeSteps;
	real initialTemperature;

	real density;
	Box box;
	bool useNN;
	PBCTYPE pbcType;
	real scale;

	bool saveTrajectory;
	int trajectorySteps;

	real vvMax, clusterKinEneSum, kinEneSum, potEnergy, bondedEnergy,
		feaEnergy, virial, feaVirial, clusterVirial, temperature;
	real wallPressure;
	int averageSteps;
	void zeroCurrVals(){
		vvMax = kinEneSum = potEnergy = bondedEnergy = 0.0;
		feaEnergy = virial = temperature = wallPressure = 0.0;
		feaVirial = clusterVirial = clusterKinEneSum = 0;
	}

	bool adjustTemperature;
	int pairCount;

	MDSystem(int nn) {
		N = 0;
		NC = 0;
		DEV = CPU;

	}

	void init(real Lx, real Ly);
	void reset();

	void setTemperature(std::vector<int> group, int groupInd, real temp);
	void setTemperature(real temp);
	void rescaleVelocities(real vFac);
	void evalProps();

	void setParticlesPosArray(real *pos);
	void setParticlesVel(real v0);

	void setDevice(device D){
		DEV = D;
		D == CPU ? printf("System running on CPU\n") : printf("System running on GPU\n");
		if (D == GPU){
			char argc = 1;
			char **argv = NULL;
			md_device_init(argc, argv);
		}
	}

	void allocateCuda(){
		if (DEV == GPU){
			int threadsPerBlock = 256;
			dimGrid = (N + threadsPerBlock-1) / threadsPerBlock;
			dimBlock = threadsPerBlock;
			particles.allocateCuda();
			interactions.setCuda();
		}
	}

	void makeFCCBox(int n1, real L, real x, real y);
	void saveXYZ(const char* fileName, int s);
	void saveVTF(const char* fileName, int s);
	void saveClusterVTF(const char* fileName, int s);
	void saveVtk(const char* fileName);
	void saveVtkXML(const char* fileName);
	void saveVtkXMLBin(const char* fileName);

	void saveClusterXYZ(const char* fileName, int s);
	
	void saveVtkXMLBin1(const char* fileName);

	void printSystemInfo(){
		printf("Number of Particles: %d\n", N);
		printf("Number of Elastomers: %d\n", NC);
		printf("Number of bonds %d\n", (int)particles.bonds.bondList.size());
		printf("Number of Triangles %d\n", (int)particles.feaElements.tetras.size());
	}


};


#endif