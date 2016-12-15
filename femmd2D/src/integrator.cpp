///////// Integration functions //////////////////////////

#include <ctime>
#include "particles.h"
#include "md3dsystem.h"
#include "integrator.h"

void Integrator::doStep(){

	sys->steps += 1;
	sys->simTime = sys->steps*sys->dt;
	sys->zeroCurrVals();
	integratorStep1();
	if (sys->useNN)
		sys->neighborlist.update(0);
	sys->interactions.calcForces();
	integratorStep2();
	sys->props.compute();
	//sys->particles.checkOverlap();

	if (sys->saveTrajectory && sys->steps%sys->trajectorySteps == 0){
		if (sys->DEV == GPU){
			sys->particles.moveToHost();
		}
		sys->particles.feaElements.calcClusterProps();
		sys->saveClusterXYZ("traj.xyz", 1);
	}

}

void Integrator::run(int steps)
{
	if (sys->DEV == GPU){
		if (sys->start){
			sys->particles.moveToDevice();
			sys->start = false;
		}
	}
	if (sys->useNN)
		sys->neighborlist.update(1);
	zeroStuff();
	sys->interactions.calcForces();
	updateAcc();
	clock_t start1;
	double duration;
	start1 = clock();
	for (int i = 0; i < steps; i++){
		doStep();
	}
	duration = (clock() - start1) / (double)CLOCKS_PER_SEC;
	double tps = (double)steps / duration;
	printf("steps: %d time: %f; %f tps\n", steps, duration, tps);
	printf("nn updates %d %f\n", sys->neighborlist.totalUpdates, (real)steps / sys->neighborlist.totalUpdates);
	if (sys->DEV == GPU){
		sys->particles.moveToHost();
	}
}
/*
void setThermostat(real g){

	//langevin_pref1 = -langevin_gamma/time_step;
	//langevin_pref2 = sqrt(24.0*temperature*langevin_gamma/time_step);
	//p->f.f[j] = langevin_pref1*p->m.v[j]*PMASS(*p) + langevin_pref2*(d_random()-0.5)*massf;

	printf("setting thermostat, gamma=%f\n", g);
	gamma = g;

	c1 = -gamma;
	c2 = sqrt(24.0*sys->temperature*gamma / sys->dt);
}

void ApplyLangevin(){
	int n;
	real c1, c2;
	c1 = sys->thermostat.c1;
	c2 = sys->thermostat.c2;
	for (n = 0; n<sys->n_particles; n++){
		//cout << GaussianRandom() << endl;
		//mol[n].ra.x += 2*sysp.langevinNoise*(GaussianRandom()-0.5);
		//mol[n].ra.y += 2*sysp.langevinNoise*(GaussianRandom()-0.5);
		//mol[n].ra.z += 2*sysp.langevinNoise*(GaussianRandom()-0.5);

		//pa[n].x += 2*sys.langevinNoise*(genrand_real2()-0.5);
		//pa[n].y += 2*sys.langevinNoise*(genrand_real2()-0.5);
		//pa[n].z += 2*sys.langevinNoise*(genrand_real2()-0.5);

		//sys->pf[n].x += c2*(genrand_real1()-0.5) + c1 * sys->pv[n].x;
		//sys->pf[n].y += c2*(genrand_real1()-0.5) + c1 * sys->pv[n].y;
		//sys->pf[n].z += c2*(genrand_real1()-0.5) + c1 * sys->pv[n].z;

		sys->pf[n].x += c2*(d_random() - 0.5) + c1 * sys->pv[n].x;
		sys->pf[n].y += c2*(d_random() - 0.5) + c1 * sys->pv[n].y;
		sys->pf[n].z += c2*(d_random() - 0.5) + c1 * sys->pv[n].z;

	}
}
*/
/////////////////Leap Frog Steps////////////////

void Integrator::leapFrogStep1(){
	real dt = sys->dt;
	real hdt = 0.5*dt;
	//#pragma omp parallel for
	for (int i = 0; i < sys->N; i++){
		sys->particles.vel[i].x += hdt*sys->particles.acc[i].x;
		sys->particles.vel[i].y += hdt*sys->particles.acc[i].y;

		sys->particles.pos[i].x += dt*sys->particles.vel[i].x;
		sys->particles.pos[i].y += dt*sys->particles.vel[i].y;
	}
}

void Integrator::leapFrogStep2(){
	real hdt = 0.5*sys->dt;
	//#pragma omp parallel for
	for (int i = 0; i < sys->N; i++){
		sys->particles.vel[i].x += hdt*sys->particles.acc[i].x;
		sys->particles.vel[i].y += hdt*sys->particles.acc[i].y;
	}
}

void Integrator::applyPBC(){
	//#pragma omp parallel for
	for (int i = 0; i < sys->N; i++){
		applyBoundaryCondition(sys->particles.pos[i], sys->box, sys->pbcType);
	}
}

void Integrator::updateAcc(){
	//#pragma omp parallel for
	for (int i = 0; i < sys->N; i++){
		sys->particles.acc[i].x = sys->particles.force[i].x / sys->particles.mass[i];
		sys->particles.acc[i].y = sys->particles.force[i].y / sys->particles.mass[i];
	}
}

void Integrator::rescaleVelocities(real vFac){
	//#pragma omp parallel for
	for (int i = 0; i<sys->N; i++){
		sys->particles.vel[i].x *= vFac;
		sys->particles.vel[i].y *= vFac;
	}
}


void Integrator::zeroStuff(){

	sys->zeroCurrVals();
	//if (0){
	if (sys->DEV == GPU){
		zeroGPUWrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs);
	}
	//#pragma omp parallel for
	for (int i = 0; i<sys->N; i++){
		sys->particles.acc[i].x = sys->particles.acc[i].y = 0;
		sys->particles.force[i].x = sys->particles.force[i].y  = 0;
	}
	//	for(var g=0; g<atomGroups.length;g++){
	//		kinEnergy[g].val = 0;
	//		vrms[g].val = 0;
	//	}		
}

void Integrator::integratorStep1(){
	if (sys->DEV == GPU){
		integratorStep1Wrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs, sys->box, sys->dt, sys->pbcType);
	}
	else{
		leapFrogStep1();
		applyPBC();
		zeroStuff();
	}
}

void Integrator::integratorStep2(){
	if (sys->DEV == GPU){
		integratorStep2Wrapper(sys->dimGrid, sys->dimBlock, sys->particles.d_gpuArgs, sys->dt);
	}
	else{
		updateAcc();
		leapFrogStep2();
	}
}