

#ifdef _WIN32
#include <direct.h>
#elif defined __linux__
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include "md3dsystem.h"
#include "duktape.h"

/////Cython  and javascript interface/////////

MDSystem *sys;
duk_context *ctx;

////// setters ////////////

void c_setPairForce(real *epsA, real *rCutA, real *ushiftA) {
	sys->interactions.setPairForce(epsA, rCutA, ushiftA);
}

void c_setPos(real *r) { sys->setParticlesPosArray(r); }

void c_setVelv0(real v0) { sys->particles.setParticlesVel(v0); }

void c_setSpringConstant(real k) { sys->interactions.kBond = k; }

void c_setBox(real Lx, real Ly)
{
	sys->box.x = Lx; sys->box.y = Ly;
}

////// getters ////////////

int c_getNumberOfParticles() { return sys->N; }
int c_getNumberOfBonds() { return sys->particles.bonds.bondList.size() / 2; }
real* c_getPosPtr() { return (real*)(sys->particles.pos.data()); }
real* c_getVelPtr() { return (real*)(sys->particles.vel.data()); }
int* c_getBondsPtr() { return (int*)(sys->particles.bonds.bondList.data()); }
//Property* c_getEnergy(){ return (Property*)(&sys->kinEnergy); }

void c_createNNList(){
	sys->neighborlist.init();
}

void c_addParticle(real x, real y, real z, real m, real sig, int typ) {
	sys->particles.addParticle(x, y, m, sig, typ);
}

void c_addParticle1(real x, real y, real vx, real vy,
	real m, real sig, int typ) {
	sys->particles.addParticle(x, y, vx, vy, m, sig, typ);
}

void c_addElastomer(real *v, int nv, int *t, int nt, real x, real y, int typ, int g) {
	sys->particles.addElastomer(v, nv, t, nt, x, y, typ, g, 0.0, 0.0,true);
}

int c_addGroup() { return sys->particles.addGroup(); }

void c_createBond(int atomi, int atomj) { sys->particles.bonds.createBond(atomi, atomj); }

void c_zeroStuff() { sys->integrator.zeroStuff(); }

void c_calcForces() { sys->interactions.calcForces(); }

void c_doStep() { sys->integrator.doStep(); }

//void c_calcProps() { sys->calcProps(); }

void c_run(int steps) { sys->integrator.run(steps); }

void c_sysOutput(){
	//printf("%d %f %f %f\n", sys->steps, sys->kinEnergy[0].val, sys->potEnergy.val, sys->totEnergy.val);
}

//////////// duktape interface  /////////////////////////


static void dump_object(duk_context *ctx, duk_idx_t idx) {
	idx = duk_require_normalize_index(ctx, idx);

	/* The weird fn() helper is to handle lightfunc name printing (= avoid it). */
	duk_eval_string(ctx,
		"(function (o) {\n"
		"    Object.getOwnPropertyNames(o).forEach(function (k) {\n"
		"        var pd = Object.getOwnPropertyDescriptor(o, k);\n"
		"        function fn(x) { if (x.name !== 'getter' && x.name !== 'setter') { return 'func' }; return x.name; }\n"
		"        print(Duktape.enc('jx', k), Duktape.enc('jx', pd), (pd.get ? fn(pd.get) : 'no-getter'), (pd.set ? fn(pd.set) : 'no-setter'));\n"
		"    });\n"
		"})");
	duk_dup(ctx, idx);
	duk_call(ctx, 1);
	duk_pop(ctx);
}

static duk_ret_t DukFileWrite(duk_context *ctx){
	const char* filename = duk_require_string(ctx, 0);
	const char* str = duk_require_string(ctx, 1);
	std::ofstream file;
	file.open(filename);
	if (file.bad())
	{
		// Dump the contents of the file to cout.
		std::cout << "cant open file " << filename << "\n";
		file.close();
	}
	file << str;
	file.close();
	return 0;
}

static duk_ret_t DukFileAppend(duk_context *ctx){
	const char* filename = duk_require_string(ctx, 0);
	const char* str = duk_require_string(ctx, 1);
	std::ofstream file;
	file.open(filename, std::ios::app);
	if (file.bad())
	{
		// Dump the contents of the file to cout.
		std::cout << "cant open file " << filename << "\n";
		file.close();
	}
	file << str;
	file.close();
	return 0;
}

static duk_ret_t DukReadFile(duk_context *ctx){
	const char *name = duk_require_string(ctx, 0);
	int type = duk_require_int(ctx, 1);
	std::ifstream file(name);
	if (file.bad())
	{
		// Dump the contents of the file to cout.
		std::cout << file.rdbuf();
		file.close();
	}
	if (type == 0){  //int type
		std::vector<int> vtemp;
		int temp;
		while (file >> temp) {
			temp = temp;
			vtemp.push_back(temp);
		}
		file.close();
		int len = vtemp.size();
		int *tempmem = (int*)malloc(sizeof(int) * len);
		memcpy(tempmem, vtemp.data(), sizeof(int) * len);
		duk_push_external_buffer(ctx);
		duk_config_buffer(ctx, -1, (void *)tempmem, (duk_size_t)(sizeof(int) * len));
		duk_push_buffer_object(ctx, -1, 0, 4 * len, DUK_BUFOBJ_INT32ARRAY);
		return 1;
	}
	if (type == 1){ //float type
		std::vector<float> vtemp;
		float temp;
		while (file >> temp) {
			temp = temp;
			vtemp.push_back(temp);
		}
		file.close();
		int len = vtemp.size();
		float *tempmem = (float*)malloc(sizeof(float) * len);
		memcpy(tempmem, vtemp.data(), sizeof(float) * len);
		duk_push_external_buffer(ctx);
		duk_config_buffer(ctx, -1, (void *)tempmem, (duk_size_t)(sizeof(float)*len));
		duk_push_buffer_object(ctx, -1, 0, 4 * len, DUK_BUFOBJ_FLOAT32ARRAY);
		return 1;

	}
	if (type == 2){ //text type
		std::string str;
		std::string file_contents;
		printf("tring to read text file %s\n", name);
		while (std::getline(file, str))
		{
			file_contents += str;
			file_contents.push_back('\n');
		}
		file.close();
		const char *s = file_contents.c_str();
		int len = strlen(s);
		char *tempmem = (char*)malloc(sizeof(char)*len);
		memcpy(tempmem, s, len);
		duk_push_lstring(ctx, tempmem, len);
		return 1;
	}
}


static duk_ret_t DukChDir(duk_context *ctx){
	const char* name = duk_require_string(ctx, 0);
	//makeDir(name);
#ifdef _WIN32
	_chdir(name);
#elif defined __linux__
	chdir(name);
#endif

	return 0;
}

static duk_ret_t DukMkDir(duk_context *ctx){
	const char* name = duk_require_string(ctx, 0);
	//changeDir(name);
#ifdef _WIN32
	_mkdir(name);
#elif defined __linux__
	mkdir(name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
	return 0;
}


void DukRegisterFunction(duk_context *ctx, duk_c_function func, char *fname, duk_idx_t nargs){
	duk_push_global_object(ctx);
	duk_push_c_function(ctx, func, nargs /*nargs*/);
	duk_put_prop_string(ctx, -2, fname);
	duk_pop(ctx);
}

static duk_ret_t DukInitSystem(duk_context *ctx){
	printf("Init System called\n");
	real Lx = duk_require_number(ctx, 0);
	real Ly = duk_require_number(ctx, 1);
	sys->init(Lx, Ly);
	return 0;
}

static duk_ret_t DukResetSystem(duk_context *ctx){
	printf("Reset System called\n");
	sys->reset();
	return 0;
}

static duk_ret_t DukAddParticle(duk_context *ctx){
	
	real x = duk_require_number(ctx, 0);	
	real y = duk_require_number(ctx, 1);
	real vx = duk_require_number(ctx, 2);
	real vy = duk_require_number(ctx, 3);
	real m = duk_require_number(ctx, 4);
	real sig = duk_require_number(ctx, 5);
	int typ = duk_require_int(ctx, 6);
	sys->particles.addParticle(x, y, vx, vy, m, sig, typ);
	return 0;
}

static duk_ret_t DukAllocateCuda(duk_context *ctx) {
	sys->allocateCuda();
	return 0;
}

static duk_ret_t DukInitNeighborList(duk_context *ctx) {
	if (sys->useNN)
		sys->neighborlist.init();
	return 0;
}

static duk_ret_t DukRun(duk_context *ctx) {
	int steps = duk_require_int(ctx, 0);
	sys->integrator.run(steps);
	return 0;
}

static duk_ret_t DukPrintProps(duk_context *ctx) {
	printf("%d %f %f %f %f %f %f\n%f %f %f\n", 
		sys->steps, 
		sys->props.uKinetic.avgval, 
		sys->props.uPotential.avgval,
		sys->props.uTotal.avgval, 
		sys->props.virial.avgval, 
		sys->props.feaVirial.avgval,
		sys->props.pressure.avgval,
		sys->props.clusterUKinetic.avgval,
		sys->props.clusterVirial.avgval,
		sys->props.molPressure.avgval
	);
	return 0;
}

static duk_ret_t DukSaveTrajectory(duk_context *ctx) {
	bool flag = duk_require_boolean(ctx, 0);
	sys->saveTrajectory = flag;
	return 0;
}

static duk_ret_t DukSetTrajectorySteps(duk_context *ctx) {
	int traj_steps = duk_require_int(ctx, 0);
	sys->trajectorySteps = traj_steps;
	return 0;
}

static duk_ret_t DukSaveVtk(duk_context *ctx) {
	const char* name = duk_require_string(ctx, 0);
	//printf("value is a string: %s\n", name);
	sys->saveVtk(name);
	return 0;
}

static duk_ret_t DukSaveVTF(duk_context *ctx) {
	const char* name = duk_require_string(ctx, 0);
	int flag = duk_require_int(ctx, 1);
	//printf("value is a string: %s\n", name);
	sys->saveVTF(name,flag);
	return 0;
}

static duk_ret_t DukSaveClusterVTF(duk_context *ctx) {
	const char* name = duk_require_string(ctx, 0);
	int flag = duk_require_int(ctx, 1);
	//printf("value is a string: %s\n", name);
	sys->saveClusterVTF(name, flag);
	return 0;
}

static duk_ret_t DukSaveClusterXYZ(duk_context *ctx) {
	const char* name = duk_require_string(ctx, 0);
	int flag = duk_require_int(ctx, 1);
	//printf("value is a string: %s\n", name);
	sys->saveClusterXYZ(name, flag);
	return 0;
}

static duk_ret_t DukCalcClusterProps(duk_context *ctx) {
	sys->particles.feaElements.calcClusterProps();
	return 0;
}

static duk_ret_t DukSystemSetVelv0(duk_context *ctx) {
	real v0 = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	sys->particles.setParticlesVel(v0);
	return 0;
}

static duk_ret_t DukSystemSetDEV(duk_context *ctx) {
	int dev = duk_require_number(ctx, 0);
	if (dev == 0)
		sys->setDevice(CPU);
	else
		sys->setDevice(GPU);
	return 0;
}

static duk_ret_t DukSystemSetPBC(duk_context *ctx) {
	//enum PBCTYPE {NOPBC, XPBC, XYPBC};
	int pbc = duk_require_number(ctx, 0);
	if (pbc == 0)
		sys->pbcType = NOPBC;
	else if (pbc == 1)
		sys->pbcType = XPBC;
	else
		sys->pbcType = XYPBC;
	return 0;
}

static duk_ret_t DukSystemSetScale(duk_context *ctx) {
	real scale = duk_require_number(ctx, 0);
	sys->scale = scale;
	return 0;
}

static duk_ret_t DukSystemSetEquiSteps(duk_context *ctx) {
	real t = duk_require_number(ctx, 0);
	sys->props.step_equi = t;
	return 0;
}

static duk_ret_t DukSystemSetAvgSteps(duk_context *ctx) {
	real t = duk_require_number(ctx, 0);
	sys->props.step_avg = t;
	return 0;
}

static duk_ret_t DukSystemSetAvgCountRdf(duk_context *ctx) {
	real t = duk_require_number(ctx, 0);
	sys->props.avgCountRdf = t;
	return 0;
}

static duk_ret_t DukSystemSetResetProps(duk_context *ctx) {
	sys->props.reset();
	return 0;
}

static duk_ret_t DukSystemSetRdfRange(duk_context *ctx) {
	real t = duk_require_number(ctx, 0);
	sys->props.rangeRdf = t;
	return 0;
}

static duk_ret_t DukSystemSet_dt(duk_context *ctx) {
	real dt = duk_require_number(ctx, 0);
	sys->dt = dt;
	return 0;
}

static duk_ret_t DukSystemGet_dt(duk_context *ctx) {
	duk_push_number(ctx, sys->dt);
	return 1;
}

static duk_ret_t DukSystemSetBox(duk_context *ctx) {
	real lx = duk_require_number(ctx, 0); 
	real ly = duk_require_number(ctx, 1); 
	sys->box.x = lx;
	sys->box.y = ly;
	return 0;
}

static duk_ret_t DukSystemGetBox(duk_context *ctx) {
	duk_idx_t obj_idx;
	obj_idx = duk_push_object(ctx);
	duk_push_number(ctx, sys->box.x);
	duk_put_prop_string(ctx, obj_idx, "x");
	duk_push_number(ctx, sys->box.y);
	duk_put_prop_string(ctx, obj_idx, "y");
	return 1;
}

static duk_ret_t DukSystemSetNN(duk_context *ctx) {
	real dt = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	sys->dt = dt;
	return 0;
}

static duk_ret_t DukSystemSetNumTypes(duk_context *ctx) {
	real nt = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	sys->particles.numTypes = nt;
	return 0;
}

static duk_ret_t DukSystemSetGravity(duk_context *ctx) {
	real g = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	if (g > 0){
		sys->interactions.gravity = g;
		sys->interactions.gravityForce = true;
	}
	else
		sys->interactions.gravityForce = false;
	return 0;
}

static duk_ret_t DukSystemSetKBond(duk_context *ctx) {
	real k = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	if (k > 0){
		sys->interactions.kBond = k;
		sys->interactions.bondForces = true;
	}
	else
		sys->interactions.bondForces = false;
	return 0;
}

static duk_ret_t DukSystemSetKArea(duk_context *ctx) {
	real k = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	if (k > 0){
		sys->interactions.kArea = k;
		sys->interactions.areaForces = true;
	}
	else
		sys->interactions.areaForces = false;
	return 0;
}

static duk_ret_t DukSystemSetYoungsModulus(duk_context *ctx) {
	real E = duk_require_number(ctx, 0);
	//printf("v0 %f\n", v0);
	if (E > 0){
		sys->interactions.E = E;
		sys->interactions.feaForces = true;
	}
	else
		sys->interactions.feaForces = false;
	sys->interactions.setElasticConstants();
	return 0;
}


static duk_ret_t DukSystemSetPairPotential(duk_context *ctx) {

	duk_size_t sz;
	real *eps;
	double* teps = (double*)duk_require_buffer_data(ctx, 0, &sz);
	int neps = sz / sizeof(double);
	if (sizeof(double) != sizeof(real)){
		eps = (real*)malloc(neps*sizeof(real));
		for (int i = 0; i < neps; i++) eps[i] = teps[i];
	}
	else{
		eps = (real*)teps;
	}
	real *rc;
	double* trc = (double*)duk_require_buffer_data(ctx, 1, &sz);
	int nrc = sz / sizeof(double);
	if (sizeof(double) != sizeof(real)){
		rc = (real*)malloc(nrc*sizeof(real));
		for (int i = 0; i < nrc; i++) rc[i] = trc[i];
	}
	else{
		rc = (real*)trc;
	}
	real *ushift;
	double* tushift = (double*)duk_require_buffer_data(ctx, 2, &sz);
	int nushift = sz / sizeof(double);
	if (sizeof(double) != sizeof(real)){
		ushift = (real*)malloc(nushift*sizeof(real));
		for (int i = 0; i < nushift; i++) ushift[i] = tushift[i];
	}
	else{
		ushift = (real*)tushift;
	}
	//for (int i = 0; i < neps; i++){
	//	printf("xx %f %f %f\n", eps[i], rc[i], ushift[i]);
	//}
	sys->interactions.setPairForce(eps, rc, ushift);
	return 0;
}

static duk_ret_t  DukAddElastomer(duk_context *ctx){

	duk_size_t sz;
	float* v = (float*)duk_require_buffer_data(ctx, 0, &sz);
	//int nv = duk_require_number(ctx, 1);
	int nv = sz / (2 * sizeof(float));
	real* vert;
	if (sizeof(float) != sizeof(real)){
		vert = (real*)malloc(2*nv*sizeof(real));
		for (int i = 0; i < 2 * nv; i++) vert[i] = v[i];
	}
	else{
		vert = (real*)v;
	}
	int* cells = (int*)duk_require_buffer_data(ctx, 1, &sz);
	//int nc = duk_require_number(ctx, 3);
	int nc = sz / (3 * sizeof(int));

	real x = duk_require_number(ctx, 2);
	real y = duk_require_number(ctx, 3);

	int typ = duk_require_int(ctx, 4);
	int g = duk_require_int(ctx, 5);
	real sig = duk_require_number(ctx, 6);
	real mass = duk_require_number(ctx, 7);
	bool exc = duk_require_boolean(ctx, 8);

	//sys->particles.addElastomer(vtemp.data(), vtemp.size() / 2, temp_t.data(), temp_t.size() / 3, c.x, c.y, 0, 0, 1.0, N, true);
	sys->particles.addElastomer(vert, nv, cells, nc, x, y, typ, g, sig, mass, exc);
	return 0;
}

static duk_ret_t DukSystemInfo(duk_context *ctx) {
	duk_idx_t info_obj;
	info_obj = duk_push_object(ctx);

	duk_push_number(ctx, sys->N);
	duk_put_prop_string(ctx, info_obj, "N");

	duk_push_number(ctx, sys->NC);
	duk_put_prop_string(ctx, info_obj, "nelastomers");

	duk_push_number(ctx, sys->particles.bonds.bondList.size());
	duk_put_prop_string(ctx, info_obj, "nbonds");

	duk_push_number(ctx, sys->particles.feaElements.tetras.size());
	duk_put_prop_string(ctx, info_obj, "ntrigs");

	duk_push_number(ctx, sys->dt);
	duk_put_prop_string(ctx, info_obj, "dt");

	duk_push_number(ctx, sys->interactions.kBond);
	duk_put_prop_string(ctx, info_obj, "kBonds");

	duk_push_number(ctx, sys->interactions.kArea);
	duk_put_prop_string(ctx, info_obj, "kArea");

	duk_push_number(ctx, sys->interactions.E);
	duk_put_prop_string(ctx, info_obj, "E");

	duk_idx_t bobj = duk_push_object(ctx);
	duk_push_number(ctx, sys->box.x);
	duk_put_prop_string(ctx, bobj, "x");
	duk_push_number(ctx, sys->box.y);
	duk_put_prop_string(ctx, bobj, "y");
	duk_put_prop_string(ctx, info_obj, "box");
	return 1;
}
/*
Prop temperature;
Prop uPotential;
Prop uKinetic;*
Prop clusterUKinetic;
Prop uTotal;
Prop uBonded;
Prop uFea;
Prop pressure;
Prop molPressure;
Prop virial;
Prop feaVirial;
Prop clusterVirial;
*/

#define ADDPROP(PROP,QPROP)\
bobj = duk_push_object(ctx);\
duk_push_number(ctx, sys->props.PROP.avgval);\
duk_put_prop_string(ctx, bobj, "avgval");\
duk_push_number(ctx, sys->props.PROP.val);\
duk_put_prop_string(ctx, bobj, "curr");\
duk_push_number(ctx, sys->props.PROP.stdv);\
duk_put_prop_string(ctx, bobj, "stdv");\
duk_put_prop_string(ctx, props_obj, QPROP)\


static duk_ret_t DukSystemGetProps(duk_context *ctx) {
	duk_idx_t props_obj;
	props_obj = duk_push_object(ctx);

	duk_push_number(ctx, sys->steps);
	duk_put_prop_string(ctx, props_obj, "steps");
	duk_idx_t bobj;
	ADDPROP(uKinetic,"uKinetic");
	ADDPROP(uPotential,"uPotential");
	ADDPROP(uFea,"uFea");
	ADDPROP(uBonded,"uBonded");
	ADDPROP(uTotal,"uTotal");
	ADDPROP(pressure,"pressure");
	ADDPROP(virial,"virial");
	ADDPROP(feaVirial,"feaVirial");

	return 1;
}


void registerMDFunctions(duk_context *ctx){
	DukRegisterFunction(ctx, DukFileWrite, "write_to_file", 2);
	DukRegisterFunction(ctx, DukFileAppend, "append_to_file", 2);
	DukRegisterFunction(ctx, DukReadFile, "read_file", 2);
	DukRegisterFunction(ctx, DukMkDir, "make_dir", 1);
	DukRegisterFunction(ctx, DukChDir, "change_dir", 1);
}


static duk_function_list_entry system_funcs[] = {
	{ "init", DukInitSystem, 2 },
	{ "reset", DukResetSystem, 0 },
	{ "add_particle", DukAddParticle, 7},
	{ "add_elastomer", DukAddElastomer, 9 },
	{ "allocate_cuda", DukAllocateCuda, 0 },
	{ "init_neighbor_list",DukInitNeighborList, 0 },
	{ "set_pair_potential", DukSystemSetPairPotential,3 },
	{ "save_vtk", DukSaveVtk, 1 },
	{ "save_vtf", DukSaveVTF, 2 },
	{ "calc_cluster_props", DukCalcClusterProps, 0 },
	{ "save_cluster_vtf", DukSaveClusterVTF, 2 },
	{ "save_cluster_xyz", DukSaveClusterXYZ, 2 },
	{ "save_trajectory", DukSaveTrajectory, 1 },
	{ "set_trajectory_steps", DukSetTrajectorySteps, 1 },
	{ "run", DukRun, 1 },

	{ "print_props", DukPrintProps, 0 },
	{ "get_props", DukSystemGetProps, 0 },
	{ "info", DukSystemInfo, 0 },
	{ "set_velocities", DukSystemSetVelv0, 1 },
	{ "set_dt", DukSystemSet_dt, 1 },
	{ "set_scale", DukSystemSetScale, 1 },
	{ "set_equilibrium_steps", DukSystemSetEquiSteps, 1 },
	{ "set_average_steps_rdf", DukSystemSetAvgCountRdf, 1 },
	{ "set_average_average_steps", DukSystemSetAvgSteps, 1 },
	{ "set_range_rdf", DukSystemSetRdfRange, 1 },
	{ "set_reset_properties", DukSystemSetResetProps, 1 },
	{ "get_dt", DukSystemGet_dt, 0 },
	{ "set_box", DukSystemSetBox, 2 },
	{ "get_box", DukSystemGetBox, 0 },
	{ "useNN", DukSystemSetNN, 1 },
	{ "set_numTypes", DukSystemSetNumTypes, 1 },
	{ "set_gravity", DukSystemSetGravity, 1 },
	{ "set_kBond", DukSystemSetKBond, 1 },
	{ "set_kArea", DukSystemSetKArea, 1 },
	{ "set_youngs_modulus", DukSystemSetYoungsModulus, 1 },
	{ "DEV", DukSystemSetDEV, 1 },
	{ "PBC", DukSystemSetPBC, 1 },
	{ NULL, NULL, 0 }
};


void registerSystemObject(duk_context *ctx) {
// Set global 'System'.
	duk_push_global_object(ctx);
	//dump_object(ctx, -1);
	duk_idx_t objid = duk_push_object(ctx);
	duk_put_function_list(ctx, -1, system_funcs);
	duk_put_prop_string(ctx, -2, "System");
	duk_pop(ctx);
	//printf("top after pop: %ld\n", (long)duk_get_top(ctx));
	
}


void runInterpreter(int argc, char *argv[]){
	duk_context *ctx = NULL;

	if (argc < 2) {
		fprintf(stderr, "Usage: md <test.js>\n");
		fflush(stderr);
		exit(1);
	}

	ctx = duk_create_heap_default();
	if (!ctx) {
		printf("Failed to create a Duktape heap.\n");
		exit(1);
	}
	duk_eval_string(ctx, "print('md js interpreter running');");
	duk_eval_string(ctx, "print('Duktape version is ' + Duktape.version);");
	printf("config file: %s\n", argv[1]);
	duk_eval_string(ctx, "var DEV={CPU:0,GPU:1};");
	duk_eval_string(ctx, "var PBCTYPE = {NOPBC:0, XPBC:1, XYPBC:2};");
	duk_eval_string(ctx, "var FILECONTENT = { INT:0, FLOAT: 1, TEXT: 2}");

	registerMDFunctions(ctx);
	sys = new MDSystem(0);
	registerSystemObject(ctx);

	duk_push_global_object(ctx);
	//duk_dump_context_stdout(ctx);
	if (duk_peval_file(ctx, argv[1]) != 0) {
		printf("Error: %s\n", duk_safe_to_string(ctx, -1));
		goto finished;
	}
	duk_pop(ctx);  /* ignore result */
	
	duk_get_prop_string(ctx, -1, "mdmain");
	if (duk_pcall(ctx, 0) != 0) {
		printf("Error: %s\n", duk_safe_to_string(ctx, -1));
	}
	duk_pop(ctx);  /* ignore result */

	//duk_dump_context_stdout(ctx);

finished:
	duk_destroy_heap(ctx);
	exit(0);
}
