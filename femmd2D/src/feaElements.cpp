#include <vector>
#include <algorithm>
#include "md3dsystem.h"
#include "feaElements.h"



void FeaElements::initElements(){
	offset = 0; //FIXME
	offset = clusters[0].offset;
	printf("Num of triangles: %d\n", numTetras);
	d_refPos = refPos;
	d_tetras = tetras;
	d_xm = xm;
	d_refVol = refVol;
	d_refPos_ptr = thrust::raw_pointer_cast(d_refPos.data());
	d_tetras_ptr = thrust::raw_pointer_cast(d_tetras.data());
	d_xm_ptr = thrust::raw_pointer_cast(d_xm.data());
	d_refVol_ptr = thrust::raw_pointer_cast(d_refVol.data());
}

void FeaElements::reset(){
	clusters.clear();
	clusters.resize(0);

	refPos.clear();
	unfoldedPos.clear();
	tetras.clear();
	xm.clear();
	refVol.clear();
	currVol.clear();

	//d_refPos.clear();
	clearDevVector(d_refPos);
	//d_tetras.clear();
	clearDevVector(d_tetras);
	//d_xm.clear();
	clearDevVector(d_xm);
	//d_refVol.clear();
	clearDevVector(d_refVol);
	numTetras = 0;
	offset = 0;
	tetraListLen = 0;
	tetraNN.clear();
	//d_tetraNN.clear();
	clearDevVector(d_tetraNN);
	offset = 0;
}

void FeaElements::checkAreas(VectorR box) {
	VectorR r[3], r0[3];

	for (int it = 0; it < tetras.size(); it++){
		for (int i = 0; i < 3; i++) {
			int pi = tetras[it][i];
			r[i].x = particles->pos[pi].x;
			r[i].y = particles->pos[pi].y;
			r0[i].x = refPos[pi].x;
			r0[i].y = refPos[pi].y;
		}

		real dx01 = r[0].x - r[1].x;
		if (dx01 > box.x / 2.0) r[1].x += box.x;
		if (dx01 < -box.x / 2.0) r[1].x -= box.x;

		real dx02 = r[0].x - r[2].x;
		if (dx02 > box.x / 2.0) r[2].x += box.x;
		if (dx02 < -box.x / 2.0) r[2].x -= box.x;

		real dy01 = r[0].y - r[1].y;
		if (dy01 > box.y / 2.0) r[1].y += box.y;
		if (dy01 < -box.y / 2.0) r[1].y -= box.y;

		real dy02 = r[0].y - r[2].y;
		if (dy02 > box.y / 2.0) r[2].y += box.y;
		if (dy02 < -box.y / 2.0) r[2].y -= box.y;

		//real vol = 0.5*((r[1].x*r[2].y - r[2].x*r[1].y) - (r[2].x*r[0].y-r[0].x*r[2].y) - (r[1].x*r[2].y-r[2].x*r[1].y));
		//real vol = 0.5*((r0[1].x*r0[2].y - r0[2].x*r0[1].y) - (r0[2].x*r0[0].y - r0[0].x*r0[2].y) - (r0[1].x*r0[2].y - r0[2].x*r0[1].y));
		real vol = 0.5*((r0[2].y - r0[0].y)*(r0[1].x - r0[0].x) - (r0[1].y - r0[0].y)*(r0[2].x - r0[0].x));
		real newVol = 0.5*((r[2].y - r[0].y)*(r[1].x - r[0].x) - (r[1].y - r[0].y)*(r[2].x - r[0].x));
		if (vol*newVol < 0)printf("Error in triangle %d %f %f \n", it, newVol, vol);
	}
}

void FeaElements::unfoldPos(VectorR box){
	for (int ei = 0; ei < clusters.size(); ei++){
		real xmin = 100000;
		real xmax = -100000;
		real ymin = 100000;
		real ymax = -100000;
		real xc = 0;
		real yc = 0;
		int eoffset = clusters[ei].offset;
		int esize = clusters[ei].nvertices;
		for (int pi = 0; pi < esize; pi++){
			unfoldedPos[pi + eoffset] = particles->pos[pi + eoffset];
			xc += unfoldedPos[pi + eoffset].x;
			yc += unfoldedPos[pi + eoffset].y;
			xmin = std::min(xmin, unfoldedPos[pi + eoffset].x);
			xmax = std::max(xmax, unfoldedPos[pi + eoffset].x);
			ymin = std::min(ymin, unfoldedPos[pi + eoffset].y);
			ymax = std::max(ymax, unfoldedPos[pi + eoffset].y);
		}
		if (xmax - xmin > box.x / 2){
			xc = 0;
			for (int pi = 0; pi < esize; pi++){
				if (unfoldedPos[pi + eoffset].x > box.x / 2){
					unfoldedPos[pi + eoffset].x -= box.x;
				}
				xc += unfoldedPos[pi + eoffset].x;
			}
			if (xc / esize < 0){
				for (int pi = 0; pi < esize; pi++){
					unfoldedPos[pi + eoffset].x += box.x;
				}
			}
		}
		if (ymax - ymin > box.y / 2){
			yc = 0;
			for (int pi = 0; pi < esize; pi++){
				if (unfoldedPos[pi + eoffset].y > box.y / 2){
					unfoldedPos[pi + eoffset].y -= box.y;
				}
				yc += unfoldedPos[pi + eoffset].y;
			}
			if (yc / esize < 0){
				for (int pi = 0; pi < esize; pi++){
					unfoldedPos[pi + eoffset].y += box.y;
				}
			}
		}
		

		real vxc, vyc;
		xc = yc = vxc = vyc = 0;
		for (int pi = 0; pi < esize; pi++){
			//unfoldedPos[pi + eoffset] = particles->pos[pi + eoffset];
			real mpi = particles->mass[pi + eoffset];
			xc += mpi*unfoldedPos[pi + eoffset].x;
			yc += mpi*unfoldedPos[pi + eoffset].y;
			vxc += mpi*particles->vel[pi + eoffset].x;
			vyc += mpi*particles->vel[pi + eoffset].y;

		}
		clusters[ei].cmPos.x = xc / clusters[ei].mass;
		clusters[ei].cmPos.y = yc / clusters[ei].mass;
		clusters[ei].cmVel.x = vxc / clusters[ei].mass;
		clusters[ei].cmVel.y = vyc / clusters[ei].mass;
	}
}

// calculate cm using
// https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
// or  http://lammps.sandia.gov/threads/msg44589.html
// compute 1 all property / atom xs
// variable xi atom lx / 2.0 / PI*cos(c_1*2.0*PI)
// variable zeta atom lx / 2.0 / PI*sin(c_1*2.0*PI)
// compute 2 all reduce ave v_xi v_zeta
// variable xb equal lx / 2.0 / PI*(atan2(-c_2[2], -c_2[1]) + PI)

void FeaElements::updateClustersCentroid(VectorR &box){
	real twopi = 2.0*PI;
	for (int ei = 0; ei < clusters.size(); ei++){
		real cxavg = 0;
		real sxavg = 0;
		real cyavg = 0;
		real syavg = 0;
		real x, y, tx, ty;
		int eoffset = clusters[ei].offset;
		int esize = clusters[ei].nvertices;
		for (int pi = 0; pi < esize; pi++){
			x = particles->pos[pi + eoffset].x;
			tx = twopi * x / box.x;
			real ci = cos(tx);
			real si = sin(tx);
			cxavg += ci;
			sxavg += si;

			y = particles->pos[pi + eoffset].y;
			ty = twopi * y / box.y;
			ci = cos(ty);
			si = sin(ty);
			cyavg += ci;
			syavg += si;

		}
		cxavg = cxavg / esize;
		sxavg = sxavg / esize;
		real txbar = atan2(-sxavg, -cxavg) + PI;
		clusters[ei].centroid.x = box.x * txbar / twopi;
		cyavg = cxavg / esize;
		syavg = sxavg / esize;
		real tybar = atan2(-syavg, -cyavg) + PI;
		clusters[ei].centroid.y = box.y * tybar / twopi;
	}
}

void FeaElements::calcClusterProps(){
	if (clusters.size() > 0){
		unfoldPos(particles->sys->box);
		updateClustersCentroid(particles->sys->box);
	}
}
