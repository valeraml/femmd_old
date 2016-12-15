#ifdef _WIN32
#include <direct.h>
#elif defined __linux__
//#include <sys/stat.h>
#include <unistd.h>
#endif

#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "md3dsystem.h"


void runInterpreter(int argc, char *argv[]);
/*
TODO


fixed types bug
evalProps and accumProps kernels //DONE
move moveToDevice and moveToHost ouside doStep //DONE
neighbor list kernels //DONE
kernel pointers: : simplify args to kernels //DONE
bonds kernels // DONE
exclusion lists on nn and cuda //DONE
optimize kernels in do_step //DONE
add thermostat
wall kernels //DONE
periodic boundary condition functions //DONE
add boundaries conditions for top and bottom
fea kernels
add dipole
yukawa potential
cython interface
add save properties
add save file every time steps
optimize energy array in compute forces
change PBC to switch statements
*/

////////////////////////////////////////

void InitCoords(MDSystem *sys, int unitCells, real L)
{
	VectorR c, gap;
	int n, nx, ny;
	gap.x = L / unitCells;
	gap.y = L / unitCells;
	n = 0;
	for (ny = 0; ny < unitCells; ny++) {
		for (nx = 0; nx < unitCells; nx++) {
			//VSet(c, nx + 0.5, ny + 0.5);
			c.x = nx + 0.5;
			c.y = ny + 0.5;
			//VMul(c, c, gap);
			c.x = c.x*gap.x;
			c.y = c.y*gap.y;
			//VVSAdd(c, -0.5, region);
			//sys->particles.pos[n] = c;
			sys->particles.addParticle(c.x, c.y, 1.0, 1.0, 0);
			++n;
		}
	}
}

void addElesatorms(MDSystem *sys, int N){
	std::ifstream v_file("diskVertices.txt");
	if (v_file.bad())
	{
		// Dump the contents of the file to cout.
		std::cout << v_file.rdbuf();
		v_file.close();
	}

	std::vector<real> vtemp;
	real temp;
	while (v_file >> temp) {
		temp = temp;
		vtemp.push_back(temp);
	}

	std::ifstream t_file("diskTriangles.txt");
	if (t_file.bad())
	{
		// Dump the contents of the file to cout.
		std::cout << t_file.rdbuf();
		t_file.close();
	}
	std::vector<int> temp_t;
	int tempi;
	while (t_file >> tempi) {
		temp_t.push_back(tempi);
	}

	// Calculate scale factor s = 1/r0 = 1/(V0/(N*0.9*PI))
	// r0 radius of inner particles whet the disk has Radius 1
	// V0 = PI*R^2
	int Ndisk = vtemp.size() / 2;
	real r0 = sqrt(0.5*0.5 / Ndisk); // 0.75 ad hoc factor
	real scale = 0.5/r0; 
	printf("r0 %f scale %f R %f A0 %f Af %f expected A %f\n", r0, scale, scale*r0, PI*r0*r0, PI*r0*r0*scale*scale, PI*0.5*0.5);
	//scale simulation box
	sys->box.x *= scale;
	sys->box.y *= scale;
	//scale particles pos
	for (int i = 0; i < vtemp.size(); i++){
		vtemp[i] *= scale;
	}
	real L = sys->box.x;
	///sys->particles.addElastomer(vtemp.data(), vtemp.size() / 2, temp_t.data(), temp_t.size() / 3, 2*L/3, 2*L/3, 0, 0, 1.0);
	//sys->particles.addElastomer(vtemp.data(), vtemp.size() / 2, temp_t.data(), temp_t.size() / 3, 2*L/3,   L/3, 0, 0, 1.0);
	//return;

	VectorR c;
	int nx, ny;
	real b = L / sqrt(N);
	int ncells = L / b;
	for (nx = 0; nx < ncells; nx++){
		for (ny = 0; ny < ncells; ny++){
			c.x = nx + 0.5;
			c.y = ny + 0.5;
			c.x = c.x*b;
			c.y = c.y*b;
			sys->particles.addElastomer(vtemp.data(), vtemp.size() / 2, temp_t.data(), temp_t.size() / 3, c.x, c.y, 0, 0, 1.0, N, true);
		}
	}
}


int c_main(){

	//_chdir("C:\\Users\\manuel.valera\\Dropbox\\research\\cuda2\\data");
	//_chdir("C:\\tmp");

#ifdef _WIN32
	_chdir("C:\\tmp");
#elif defined __linux__
	chdir("/home/manuel/tmp");
#endif
	
	int ucells = 20;
	int N = ucells*ucells;
	//int N = 10;
	real dens = 0.5;
	real L = sqrt(N / dens); 
	real T = 0.5;
	real v0 = sqrt(2.0 * T * (1.0-1.0/N));

	MDSystem sys(0);
	sys.setDevice(CPU);
	sys.init(L, L);
	sys.particles.numTypes = 1;
	sys.dt = 0.001;

	//sys.makeFCCBox(N, L, 0, 0);
	InitCoords(&sys, ucells, L);
	//addElesatorms(&sys,N);
	sys.useNN = true;
	sys.pbcType = XYPBC;
	sys.interactions.gravity = 0;
	sys.interactions.gravityForce = false;
	sys.interactions.E = 5;
	sys.interactions.kBond = 10;
	sys.interactions.kArea = .0001;
	sys.interactions.bondForces = false;
	sys.interactions.feaForces = false;
	sys.interactions.areaForces = false;
	sys.interactions.setElasticConstants();

	real eps[] = { 1.0 };
	real rCut[] = { pow(2.0, 1.0 / 6.0) };
	rCut[0] = 3.0;
	real uShift[] = { 1.0 };
	uShift[0] = 0.0;

//	real eps[] = { 1.0, 1.0, 1.0, 1.0 };
//	real r0 = pow(2.0, 1.0 / 6.0);
//	real rCut[] = { r0, r0, r0, r0 };
//	real uShift[] = { 1.0, 1.0, 1.0, 1.0 };

	sys.interactions.setPairForce(eps, rCut, uShift);

	sys.particles.setParticlesVel(v0);
	sys.allocateCuda();
	if(sys.useNN) 
		sys.neighborlist.init();
	//sys.particles.checkOverlap();
	//sys.setTemperature(sys.atomGroups[0],T);
	sys.saveVTF("movie.vtf", 0);
	//sys.saveVtk("movie");
	//sys.saveXYZ("movie.xyz", 0);
	//return 0;
	sys.start = true;
	//for (int i = 0; i < 10; i++){
		//sys.setTemperature(T);
		//sys.integrator.run(100);
	//}
	for (int i = 0; i<10000; i++){
		sys.integrator.run(10000);
		printf("%d %f %f %f %f %f\n", sys.steps, sys.props.uKinetic.avgval, sys.props.uPotential.avgval, 
			sys.props.uTotal.avgval, sys.props.virial.avgval, sys.props.pressure.avgval);
		sys.saveVTF("movie.vtf", 1);
		//sys.saveXYZ("movie.xyz", 1);
		//sys.saveVtk("movie");
		//sys.saveVtkXML("movieVTU");
		//sys.saveVtkXMLBin("movieVTUBin");
		//sys.saveVtkXMLBin1("testVTUBin");
	}
	return 0;

}

int js_main(int argc, char *argv[]) {
	runInterpreter(argc, argv);
	return 0;
}


int main(int argc, char *argv[]) {
#ifdef _WIN32
	//_chdir("C:\\tmp");
#elif defined __linux__
	//chdir("/home/manuel/tmp");
#endif
	//return c_main();
	return js_main(argc,argv);
}

