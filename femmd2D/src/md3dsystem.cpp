// JavaScript Document

/*
	A 3D molecular dynamics simulation in c/c++
	
	Copyright 2015, Manuel Valera
	
	Permission is hereby granted, free of charge, to any person obtaining a copy of 
	this software and associated data and documentation (the "Software"), to deal in 
	the Software without restriction, including without limitation the rights to 
	use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
	of the Software, and to permit persons to whom the Software is furnished to do 
	so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all 
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
	PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
	ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
	OTHER DEALINGS IN THE SOFTWARE.

	Except as contained in this notice, the name of the author shall not be used in 
	advertising or otherwise to promote the sale, use or other dealings in this 
	Software without prior written authorization.
	
	
*/


#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "md3dsystem.h"

real PI;

//////////// Class system functions ///////////////////////////

//////////// Inititalization functions///////////////


void MDSystem::init(real Lx, real Ly){
	PI = 4.0 * atan(1.0);
	start = true;
	steps = 0;
	dt = 0.005;
	simTime = 0;
	thermalizeSteps = 1000;
	averageSteps = 100;
	initialTemperature = 1.0;
	pbcType = XYPBC;

	density = 0;
	useNN = true;

	//Properties 

	saveTrajectory = false;
	trajectorySteps = 100;
	bool adjustTemperature = true;

	box.x = Lx;
	box.y = Ly; 
	density = N / (Lx*Ly);
	
	particles.init(this);
	neighborlist.init(this);
	interactions.init(this);
	integrator.init(this);
	props.sys = this;

	particles.numTypes = 1;
}

void MDSystem::reset(){
	N = 0;
	NC = 0;
	particles.reset();
	neighborlist.reset();
	interactions.reset();
	props.reset();
}


void MDSystem::setTemperature(std::vector<int> group, int groupInd, real temp){

	real vvSum = 0;
	real kinEne = 0;
	real vFac;
	real mass = 1;
	int len = particles.partGroupLen[groupInd];
	
	for(int i=0; i<len; i++){
		int pi  = group[i];
		particles.vv[pi] = particles.vel[pi].x*particles.vel[pi].x + 
			particles.vel[pi].y*particles.vel[pi].y;
		vvSum += particles.vv[pi];
		kinEne += 0.5f*this->particles.mass[pi] * particles.vv[pi];
	}
	//var vMag = Math.sqrt(2*temp/mass); 
	//vFac = vMag/Math.sqrt(vvSum/numAtoms);
	real currTemp = 2 * kinEne / (3 * N);
	vFac = sqrt(temp/currTemp);
	for(int i=0; i<len; i++){
		int pi = group[i];
		particles.vel[pi].x *= vFac;
		particles.vel[pi].y *= vFac;
	}
}

void MDSystem::setTemperature(real temp){

	real vvSum = 0;
	real kinEne = 0;
	real vFac;

	for (int i = 0; i<N; i++){
		int pi = i;
		particles.vv[pi] = particles.vel[pi].x*particles.vel[pi].x +
			particles.vel[pi].y*particles.vel[pi].y;
		vvSum += particles.vv[pi];
		kinEne += 0.5f*this->particles.mass[pi] * particles.vv[pi];
	}
	//var vMag = Math.sqrt(2*temp/mass); 
	//vFac = vMag/Math.sqrt(vvSum/numAtoms);
	real currTemp = kinEne / N;
	vFac = sqrt(temp / currTemp);
	for (int i = 0; i<N; i++){
		int pi = i;
		particles.vel[pi].x *= vFac;
		particles.vel[pi].y *= vFac;
	}
	//particles.zeroCM();
}


real findMax(dvector<real> &v);
void squareVec(dvector<VectorR> &v, dvector<real> &vv);
void multVec(dvector<real> &vv, dvector<real> &m);
real reduce(dvector<real> &vv);

extern real tempKinEneVVMax[2];

void MDSystem::evalProps(){

	if (DEV == GPU){
		//squareVec(particles.d_vel, particles.d_vv);
		//vvMax = findMax(particles.d_vv);
		//multVec(particles.d_vv, particles.d_mass);
		//kinEneSum[0] = reduce(particles.d_vv);
		kinEneSum = tempKinEneVVMax[0];
		vvMax = tempKinEneVVMax[1];
	}
	else{
		for (int pi = 0; pi<N; pi++){
			particles.vv[pi] = particles.vel[pi].x*particles.vel[pi].x + particles.vel[pi].y*particles.vel[pi].y;
			vvMax = std::max(vvMax, particles.vv[pi]);
			real kin = particles.mass[pi] * particles.vv[pi];
			//vvSum += particles.vv[pi];
			kinEneSum += kin;
		}
		if (particles.feaElements.clusters.size()>0){
			for (int ci = 0; ci < particles.feaElements.clusters.size(); ci++){
				real cvv = particles.feaElements.clusters[ci].cmVel.x*particles.feaElements.clusters[ci].cmVel.x +
					particles.feaElements.clusters[ci].cmVel.y*particles.feaElements.clusters[ci].cmVel.y;
				clusterKinEneSum += particles.feaElements.clusters[ci].mass* cvv;
			}
		}
	}
	neighborlist.dispHi += sqrt(vvMax) * dt;

}


/////////// Initialization Functions ///////////////////////////


void MDSystem::setParticlesPosArray(real* posIn) {
	for (int p = 0; p < N; p++) {
		particles.pos[p].x = posIn[3 * p];
		particles.pos[p].y = posIn[3 * p + 1];
	}
}

void MDSystem::makeFCCBox(int n1, real L, real x0, real y0){

	int c, i, j, k, m, p;
	real b, vSum[3] = { 0.0, 0.0 };
	real rFCC[2][2] = { { 0.0, 0.0}, { 0.5, 0.5 } };
	real rCell[2];
	int numberOfParticles = n1;

	// compute length width height of box
	//L = pow(n_articles / density, 1.0 / 3.0);
	//for (i = 0; i < 3; i++)
	//box[i] = L;
	real delta = L / 100;
	L = L - delta;
	real totx = 0;

	// use face centered cubic (fcc) lattice for initial positions
	// find number c of unit cells needed to place all particles
	for (c = 1;; c++)
		if (2* c*c >= numberOfParticles)
			break;
	b = L / c;			// side of unit cell
	p = 0;			// particles placed so far
	for (i = 0; i < c; i++) {
		rCell[0] = i * b;
		for (j = 0; j < c; j++) {
			rCell[1] = j * b;
			for (m = 0; m < 2; m++)	// 2 particles in cell
				if (p < numberOfParticles) {
					real x = rCell[0] + b * rFCC[m][0] + delta / 2 + x0;
					real y = rCell[1] + b * rFCC[m][1] + delta / 2 + y0;
					particles.addParticle(x, y, 1.0, 1.0, 0);
					//printf("(%f %f %f)\n",h_pos[p].x, h_pos[p].y, h_pos[p].z);
					++p;
				}
		}
	}
}


void MDSystem::saveXYZ(const char* fileName, int s) {
	FILE *dataFile;
	int i;
	if (s == 0)
		dataFile = fopen(fileName, "w");
	else
		dataFile = fopen(fileName, "a");
	fprintf(dataFile, "%d\n", N);
	fprintf(dataFile, "\n");
	//fprintf(dataFile, "box size  %f\t%f\t%f\n", box.x, box.y,1.0);
	for (i = 0; i < N; i++)
		fprintf(dataFile, "A\t%f\t%f\t%f\n", particles.pos[i].x, particles.pos[i].y, 0.0f);
	//fprintf(dataFile,"\n");
	fclose(dataFile);
}

void MDSystem::saveClusterXYZ(const char* fileName, int s) {
	FILE *dataFile;
	int i;
	if (s == 0){
		dataFile = fopen(fileName, "w");
	}
	else
		dataFile = fopen(fileName, "a");
	fprintf(dataFile, "ITEM: TIMESTEP\n%d\n", steps);
	fprintf(dataFile, "ITEM: NUMBER OF ATOMS\n%d\n", NC);
	fprintf(dataFile, "ITEM: BOX BOUNDS pp pp pp\n%f %f\n%f %f\n%f %f\n", 0.0,box.x,0.0,box.y,0.0,0.0);
	fprintf(dataFile, "ITEM: ATOMS id type x y z vx vy vz\n");
	//fprintf(dataFile, "box size  %f\t%f\t%f\n", box.x, box.y,1.0);
	for (i = 0; i < NC; i++)
		//if (bin){
		//	fwrite()
		//}
		//else{
			fprintf(dataFile, "%d %d %f %f %f %f %f %f\n",
				i, particles.type[i],
				particles.pos[i].x, particles.pos[i].y, 0.0f,
				particles.vel[i].x, particles.vel[i].y, 0.0f);
		//}
	//fprintf(dataFile,"\n");
	fclose(dataFile);
}

void MDSystem::saveVtk(const char* fileName) {
	FILE *dataFile;
	int i;
	if (NC == 0) return;
	char name[255];
	sprintf(name, "%s%.9d.vtk", fileName,steps);
	dataFile = fopen(name, "w");
	fprintf(dataFile,"# vtk DataFile Version 3.1\n");
	fprintf(dataFile,"Really cool data\n");
	fprintf(dataFile, "ASCII\n");
	fprintf(dataFile, "DATASET UNSTRUCTURED_GRID\n\n");
	fprintf(dataFile, "POINTS %d FLOAT\n",N);
	particles.feaElements.unfoldPos(box);
	for (i = 0; i < N; i++)
		//fprintf(dataFile, "%f %f %f\n", particles.pos[i].x, particles.pos[i].y, 0.0f);
		fprintf(dataFile, "%f %f %f\n", particles.feaElements.unfoldedPos[i].x, particles.feaElements.unfoldedPos[i].y, 0.0f);
	fprintf(dataFile, "\n");
	if (particles.feaElements.tetras.size() > 0){
		int ntet = particles.feaElements.tetras.size();
		fprintf(dataFile, "CELLS %d %d\n",ntet, 4*ntet);
		for (i = 0; i < ntet; i++){
			tetraIndexes vert = particles.feaElements.tetras[i];
			fprintf(dataFile, "%d %d %d %d\n", 3, vert[0], vert[1], vert[2]);
		}
		fprintf(dataFile, "\n");
		fprintf(dataFile, "CELL_TYPES %d\n",ntet);
		int VTK_TRIANGLE = 5;
		int VTK_TETRA = 10;
		for (i = 0; i < ntet; i++)
			fprintf(dataFile, "%d ", VTK_TRIANGLE);
		fprintf(dataFile,"\n\n");

	}	
	fprintf(dataFile, "POINT_DATA %d\n", N);
	fprintf(dataFile, "SCALARS VelocitySquare FLOAT\n");
	fprintf(dataFile, "LOOKUP_TABLE default\n");
	for (i = 0; i < N; i++)
		fprintf(dataFile, "%f\n", particles.vv[i]);
	fprintf(dataFile,"\n\n");

	fprintf(dataFile, "VECTORS Velocity FLOAT\n");
	for (i = 0; i < N; i++)
		fprintf(dataFile, "%f %f %f\n", particles.vel[i].x, particles.vel[i].y, 0.0f);

	//fprintf(dataFile, "POINT_DATA %d\n", N);
	//fprintf(dataFile, "\n");
	//fprintf(dataFile, "box size  %f\t%f\t%f\n", box.x, box.y,1.0);

	//fprintf(dataFile,"\n");
	fclose(dataFile);
}




void MDSystem::saveVTF(const char* fileName, int s) {
	FILE *dataFile;
	char typName[] = { 'A', 'B', 'C', 'D', 'E', 'F' };
	int i;
	if (s == 0){
		dataFile = fopen(fileName, "w");
		for (int i = 0; i < N; i++){
			fprintf(dataFile, "atom %d type %c radius %f\n", i, typName[particles.type[i]], particles.radius[i]); //FIXME change %c to %s
		}
		if (particles.bonds.bondList.size() > 0){
			for (int b = 0; b < particles.bonds.numBonds; b++){
				fprintf(dataFile, "bond %d:%d\n", particles.bonds.bondList[2*b], particles.bonds.bondList[2*b+1]);
			}
		}
		fprintf(dataFile, "pbc %f %f %f\n", box.x, box.y, 1.0);
	}
	else
		dataFile = fopen(fileName, "a");
	fprintf(dataFile, "timestep\n");
	for (i = 0; i < N; i++)
		fprintf(dataFile, "%f %f %f\n", particles.pos[i].x, particles.pos[i].y, 0.0f);
	fclose(dataFile);
}

void MDSystem::saveClusterVTF(const char* fileName, int s) {
	FILE *dataFile;
	char typName[] = { 'A', 'B', 'C', 'D', 'E', 'F' };
	int i;
	if (s == 0){
		dataFile = fopen(fileName, "w");
		for (int i = 0; i < particles.feaElements.clusters.size(); i++){
			fprintf(dataFile, "atom %d type %c radius %f\n", i, typName[particles.type[i]], 0.5*scale+0.5); //FIXME change %c to %s
		}
		//if (particles.bonds.bondList.size() > 0){
		//	for (int b = 0; b < particles.bonds.numBonds; b++){
		//		fprintf(dataFile, "bond %d:%d\n", particles.bonds.bondList[2 * b], particles.bonds.bondList[2 * b + 1]);
		//	}
		//}
		fprintf(dataFile, "pbc %f %f %f\n", box.x, box.y, 1.0);
	}
	else
		dataFile = fopen(fileName, "a");
	fprintf(dataFile, "timestep\n");
	for (i = 0; i < particles.feaElements.clusters.size(); i++)
		fprintf(dataFile, "%f %f %f\n", particles.feaElements.clusters[i].cmPos.x, 
										particles.feaElements.clusters[i].cmPos.y, 0.0f);
	fclose(dataFile);
}


void MDSystem::saveVtkXML(const char* fileName) {
	FILE *dataFile;
	int i;
	char name[255];
	char s[255];
	sprintf(name, "%s%.9d.vtu", fileName, steps);
	dataFile = fopen(name, "w");
	fprintf(dataFile, "<?xml version=\"1.0\"?>\n");
	fprintf(dataFile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(dataFile, "\t<UnstructuredGrid>\n");
			int ntet = particles.feaElements.tetras.size();
			fprintf(dataFile, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", N, ntet);

				fprintf(dataFile, "\t\t\t<PointData Scalars=\"scalars\">\n");
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"ascii\" Name=\"VelocitySquared\">\n");
						for (i = 0; i < N; i++)
							fprintf(dataFile, "%f ", particles.vv[i]);
						fprintf(dataFile, "\n");
					fprintf(dataFile, "\t\t\t\t</DataArray>\n");
				fprintf(dataFile, "\t\t\t</PointData>\n");

				fprintf(dataFile, "\t\t\t<Points>\n");
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"3\">\n");
						for (i = 0; i < N; i++)
							fprintf(dataFile, "%f %f %f ", particles.feaElements.unfoldedPos[i].x, particles.feaElements.unfoldedPos[i].y, 0.0f);
						fprintf(dataFile, "\n");
					fprintf(dataFile, "\t\t\t\t</DataArray>\n");
				fprintf(dataFile, "\t\t\t</Points>\n");

				fprintf(dataFile, "\t\t\t<Cells>\n");
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n");
						for (i = 0; i < ntet; i++){
							tetraIndexes vert = particles.feaElements.tetras[i];
							fprintf(dataFile, "%d %d %d ", vert[0], vert[1], vert[2]);
						}
						fprintf(dataFile, "\n");
					fprintf(dataFile, "\t\t\t\t</DataArray>\n");
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" format=\"ascii\" Name=\"offsets\">\n");
						for (i = 0; i < ntet; i++)
							fprintf(dataFile, "%d ", 3*(i+1));
						fprintf(dataFile, "\n");
					fprintf(dataFile, "\t\t\t\t</DataArray>\n");
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" format=\"ascii\" Name=\"types\">\n");
						for (i = 0; i < ntet; i++)
							fprintf(dataFile, "%d ", 5);
						fprintf(dataFile, "\n");
					fprintf(dataFile, "\t\t\t\t</DataArray>\n");
				fprintf(dataFile, "\t\t\t</Cells>\n");

			fprintf(dataFile, "\t\t</Piece>\n");
		fprintf(dataFile, "\t</UnstructuredGrid>\n");
	fprintf(dataFile, "</VTKFile>");

	fclose(dataFile);
}


//<DataArray type = "Float32" Name = "foo" NumberOfComponents = "1" format = "appended" offset = "0" / >

typedef unsigned int headtyp;

void MDSystem::saveVtkXMLBin(const char* fileName) {
	FILE *dataFile;
	int i;
	char name[255];
	char s[255];
	int offset = 0;
	sprintf(name, "%s%.9d.vtu", fileName, steps);
	//printf("sizeof %d %d %d\n", sizeof(float), sizeof(int), sizeof(headtyp));
	dataFile = fopen(name, "w");
	//fprintf(dataFile, "<?xml version=\"1.0\"?>\n");
	fprintf(dataFile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(dataFile, "\t<UnstructuredGrid>\n");
			int ntet = particles.feaElements.tetras.size();
			fprintf(dataFile, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", N, ntet);
				fprintf(dataFile, "\t\t\t<PointData Scalars=\"scalars\">\n");
					offset = 0;
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"appended\" Name=\"VelocitySquared\" offset=\"%d\"/>\n",offset);
				fprintf(dataFile, "\t\t\t</PointData>\n");

				fprintf(dataFile, "\t\t\t<Points>\n");
					offset += N * sizeof(float) + sizeof(headtyp);
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"appended\" NumberOfComponents=\"3\" offset=\"%d\"/>\n",offset);
				fprintf(dataFile, "\t\t\t</Points>\n");
				
				fprintf(dataFile, "\t\t\t<Cells>\n");
					offset += 3 * N * sizeof(float) + sizeof(headtyp);
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"appended\" offset=\"%d\"/>\n",offset);
					offset += ntet * 3 * sizeof(int) + sizeof(headtyp);
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" Format=\"appended\" offset=\"%d\"/>\n",offset);
					offset += ntet*sizeof(int) + sizeof(headtyp);
					fprintf(dataFile, "\t\t\t\t<DataArray type=\"UInt8\" Name=\"types\" Format=\"appended\" offset=\"%d\"/>\n",offset);
				fprintf(dataFile, "\t\t\t</Cells>\n");
				
			fprintf(dataFile, "\t\t</Piece>\n");
		fprintf(dataFile, "\t</UnstructuredGrid>\n");
		fprintf(dataFile, "\t<AppendedData encoding=\"raw\">\n"); 
		fprintf(dataFile, "\t_");

		headtyp size = N * sizeof(float);
		const char* p = reinterpret_cast<const char*>(&size);
		//unsigned char *p = (unsigned char*)&size;
		fwrite(p, sizeof(headtyp), 1, dataFile);
		for (i = 0; i < N; i++){
			float temp = particles.vv[i];
			fwrite((void*)&temp, sizeof(float), 1, dataFile);
		}
		size = N * 3 * sizeof(float);
		fwrite(p, sizeof(headtyp), 1, dataFile);
		for (i = 0; i < N; i++){
			float r[3];
			r[0] = particles.feaElements.unfoldedPos[i].x;
			r[1] = particles.feaElements.unfoldedPos[i].y;
			r[2] = 0.0f;
			fwrite((void*)r, sizeof(float), 3, dataFile);
		}
		
		size = 3 * ntet*sizeof(int);
		fwrite(p, sizeof(headtyp), 1, dataFile);
		for (i = 0; i < ntet; i++){
			tetraIndexes vert = particles.feaElements.tetras[i];
			int v[3];
			v[0] = (int)vert[0];
			v[1] = (int)vert[1];
			v[2] = (int)vert[2];
			fwrite((void*)v, sizeof(int), 3, dataFile);
		}
		size = ntet*sizeof(int);
		fwrite(p, sizeof(headtyp), 1, dataFile);
		for (i = 0; i < ntet; i++){
			unsigned int temp1 = 3 * (i + 1);
			fwrite((void*)&temp1, sizeof(int), 1, dataFile);
		}
		size = ntet*sizeof(char);
		fwrite(p, sizeof(headtyp), 1, dataFile);
		for (i = 0; i < ntet; i++){
			unsigned char temp = 5;
			//fputs("5", dataFile);
			fwrite(&temp, sizeof(unsigned char), 1, dataFile);
		}
		
		fprintf(dataFile,"\n\t</AppendedData>\n");
	fprintf(dataFile, "</VTKFile>");

	fclose(dataFile);
}


void MDSystem::saveVtkXMLBin1(const char* fileName) {
	FILE *dataFile;
	int i;
	char name[255];
	char s[255];
	int offset = 0;
	sprintf(name, "%s%.9d.vtu", fileName, steps);
	//printf("sizeof %d %d %d\n", sizeof(float), sizeof(int), sizeof(headtyp));
	dataFile = fopen(name, "w");
		int ntet = 2;
		int N1 = 4;
		VectorR points[4];
		points[0].x = 0.0;
		points[0].y = 0.0;
		points[1].x = 1.0;
		points[1].y = 0.0;
		points[2].x = 0.0;
		points[2].y = 1.0;
		points[3].x = 1.0;
		points[3].y = 1.0;
		tetraIndexes tets[2];
		tets[0][0] = 0;
		tets[0][1] = 1;
		tets[0][2] = 2;
		tets[1][0] = 1;
		tets[1][1] = 2;
		tets[1][2] = 3;
		real vv[] = { 0.1, 0.2, 0.3, 0.4 };

		bool ascii = false;
		if (ascii){
			fprintf(dataFile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
			fprintf(dataFile, "\t<UnstructuredGrid>\n");
			fprintf(dataFile, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", N1, ntet);

			fprintf(dataFile, "\t\t\t<PointData Scalars=\"scalars\">\n");
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"ascii\" Name=\"VelocitySquared\">\n");
			for (i = 0; i < N1; i++)
				fprintf(dataFile, "%f ", vv[i]);
			fprintf(dataFile, "\n");
			fprintf(dataFile, "\t\t\t\t</DataArray>\n");
			fprintf(dataFile, "\t\t\t</PointData>\n");

			fprintf(dataFile, "\t\t\t<Points>\n");
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"ascii\" NumberOfComponents=\"3\">\n");
			for (i = 0; i < N1; i++)
				fprintf(dataFile, "%f %f %f ", points[i].x, points[i].y, 0.0f);
			fprintf(dataFile, "\n");
			fprintf(dataFile, "\t\t\t\t</DataArray>\n");
			fprintf(dataFile, "\t\t\t</Points>\n");

			fprintf(dataFile, "\t\t\t<Cells>\n");
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n");
			for (i = 0; i < ntet; i++){
				tetraIndexes vert = tets[i];
				fprintf(dataFile, "%d %d %d ", vert[0], vert[1], vert[2]);
			}
			fprintf(dataFile, "\n");
			fprintf(dataFile, "\t\t\t\t</DataArray>\n");
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" format=\"ascii\" Name=\"offsets\">\n");
			for (i = 0; i < ntet; i++)
				fprintf(dataFile, "%d ", 3 * (i + 1));
			fprintf(dataFile, "\n");
			fprintf(dataFile, "\t\t\t\t</DataArray>\n");
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" format=\"ascii\" Name=\"types\">\n");
			for (i = 0; i < ntet; i++)
				fprintf(dataFile, "%d ", 5);
			fprintf(dataFile, "\n");
			fprintf(dataFile, "\t\t\t\t</DataArray>\n");
			fprintf(dataFile, "\t\t\t</Cells>\n");

			fprintf(dataFile, "\t\t</Piece>\n");
			fprintf(dataFile, "\t</UnstructuredGrid>\n");
			fprintf(dataFile, "</VTKFile>");

			fclose(dataFile);
		}
		else{

			fprintf(dataFile, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
			fprintf(dataFile, "\t<UnstructuredGrid>\n");
			fprintf(dataFile, "\t\t<Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n", N1, ntet);
			fprintf(dataFile, "\t\t\t<PointData Scalars=\"scalars\">\n");
			offset = 0;
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"appended\" Name=\"VelocitySquared\" offset=\"%d\"/>\n", offset);
			fprintf(dataFile, "\t\t\t</PointData>\n");

			fprintf(dataFile, "\t\t\t<Points>\n");
			offset += N1 * sizeof(float) + sizeof(headtyp);
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Float32\" format=\"appended\" NumberOfComponents=\"3\" offset=\"%d\"/>\n", offset);
			fprintf(dataFile, "\t\t\t</Points>\n");
			
			fprintf(dataFile, "\t\t\t<Cells>\n");
			offset += 3 * N1 * sizeof(float) + sizeof(headtyp);
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"appended\" offset=\"%d\"/>\n",offset);
			offset += ntet * 3 * sizeof(int) + sizeof(headtyp);
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" Format=\"appended\" offset=\"%d\"/>\n",offset);
			offset += ntet*sizeof(int) + sizeof(headtyp);
			fprintf(dataFile, "\t\t\t\t<DataArray type=\"UInt8\" Name=\"types\" Format=\"appended\" offset=\"%d\"/>\n",offset);
			fprintf(dataFile, "\t\t\t</Cells>\n");
			
			fprintf(dataFile, "\t\t</Piece>\n");
			fprintf(dataFile, "\t</UnstructuredGrid>\n");
			fprintf(dataFile, "<AppendedData encoding=\"raw\">\n");
			fprintf(dataFile, " _");

			headtyp size = N1 * sizeof(float);
			const char* p = reinterpret_cast<const char*>(&size);
			//unsigned char *p = (unsigned char*)&size;
			fwrite(p, sizeof(headtyp), 1, dataFile);
			for (i = 0; i < N1; i++){
				float temp = vv[i];
				fwrite((void*)&temp, sizeof(float), 1, dataFile);
			}
			size = N1 * 3 * sizeof(float);
			fwrite(p, sizeof(headtyp), 1, dataFile);
			for (i = 0; i < N1; i++){
				float r[3];
				r[0] = points[i].x;
				r[1] = points[i].y;
				r[2] = 0.0f;
				fwrite((void*)r, sizeof(float), 3, dataFile);
			}

			size = 3 * ntet*sizeof(int);
			fwrite(p, sizeof(headtyp), 1, dataFile);
			for (i = 0; i < ntet; i++){
				tetraIndexes vert = tets[i];
				int v[3];
				v[0] = (int)vert[0];
				v[1] = (int)vert[1];
				v[2] = (int)vert[2];
				fwrite((void*)v, sizeof(int), 3, dataFile);
			}
			size = ntet*sizeof(int);
			fwrite(p, sizeof(headtyp), 1, dataFile);
			for (i = 0; i < ntet; i++){
				int temp1 = 3 * (i + 1);
				fwrite((void*)&temp1, sizeof(int), 1, dataFile);
			}
			size = ntet*sizeof(char);
			fwrite(p, sizeof(headtyp), 1, dataFile);
			for (i = 0; i < ntet; i++){
				unsigned char temp = 5;
				//fputs("5", dataFile);
				fwrite(&temp, sizeof(unsigned char), 1, dataFile);
			}

			fprintf(dataFile, "</AppendedData>\n");
			fprintf(dataFile, "</VTKFile>");

			fclose(dataFile);

		}
		
}