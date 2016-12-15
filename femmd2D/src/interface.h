#include "def.h"
//#include "md3dsystem.h"
//typedef double real;
/////Cython interface/////////

//extern MDSystem *sys;

////// setters ////////////

void c_setPairForce(real *epsA, real *rCutA, real *ushiftA); 
void c_setPos(real *r);
void c_setVelv0(real v0);
void c_setSpringConstant(real k);
void c_setBox(double Lx, double Ly);


////// getters ////////////

int c_getNumberOfParticles();
int c_getNumberOfBonds();
real* c_getPosPtr();
real* c_getVelPtr();
int* c_getBondsPtr();
Property* c_getEnergy();

void c_createSystem(real Lx, real Ly); 

void c_createNNList();

void c_addParticle(real x, real y, real m, real sig, int typ); 

void c_addParticle1(real x, real y, real vx, real vy, real m, real sig, int typ);

void c_addElastomer(real *v, int nv, int *t, int nt, real x, real y, int typ, int g); 

int c_addGroup(); 

void c_createBond(int atomi, int atomj);

void c_zeroStuff();

void c_calcForces();

void c_doStep();

void c_calcProps();

void c_run(int steps);

void c_sysOutput();
