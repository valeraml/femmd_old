#ifndef __DEF_H__
#define __DEF_H__

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "vector_types.h"

//#define USE_DOUBLE

#ifdef USE_DOUBLE

typedef double real;
struct realV{ real x, y; };
struct intV{ int x, y; };
//typedef intV VectorI;
//typedef realV VectorR;
typedef int2 VectorI;
typedef double2 VectorR;

#else

typedef float real;
struct realV{real x, y;};
struct intV{ int x, y; };
//typedef intV VectorI;
//typedef realV VectorR;
typedef float2 VectorR;
typedef int2 VectorI;

#endif

extern real PI;

#define RANDOM01 ((real) rand() / (RAND_MAX))

enum device { CPU, GPU };
enum PBCTYPE {NOPBC, XPBC, XYPBC};

#define Max(x1, x2)  (((x1) > (x2)) ? (x1) : (x2))
#define Min(x1, x2)  (((x1) < (x2)) ? (x1) : (x2))

#define PropZero(v)  v.val = v.sum = v.sum2 = 0.
#define PropAccum(v)  v.sum += v.val, v.sum2 += Sqr (v.val)
#define PropAvg(v, n) \
   v.sum /= n, v.sum2 = sqrt (Max (v.sum2 / n - Sqr (v.sum), 0.))
	

void md_device_init(int argc, char *argv[]);
void md_device_report();

struct mat3x3 {
	real m[3][3];
};

struct tetraIndexes{
	int ind[3];
	inline int& operator[](int idx) { return ind[idx]; }
};

struct matNxN{
	real *data;
	real **m;
	void init(int n1){ 
		n = n1; 
		data = new real[n*n];
		m = new real*[n];
		m[0] = data;
		for (int i = 1; i < n; i++) m[i] = m[i - 1] + n;//m[i] = &data[i*n];
	}

	void clear(){
		delete[] data;
		delete[] m;
		n = 0;
	}
	real at(int i, int j){ return data[i*n + j]; }
	void set(int i, int j, real v){ data[i*n + j] = v; }
	real* operator[] (int i) { return m[i]; }
	int n;
};

typedef VectorR Box;

#define hvector thrust::host_vector
#define dvector thrust::device_vector

#define minImage(dr, box, x)                           \
	if (dr.x >= 0.5*box.x) dr.x -= box.x;              \
		else if (dr.x < -0.5*box.x) dr.x += box.x;

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif 
static void nearestImage(VectorR &dr, VectorR &box, PBCTYPE pbcType){
	switch (pbcType){
	case XYPBC:
		minImage(dr, box, x);
		minImage(dr, box, y);
		break;
	case XPBC:
		minImage(dr, box, x);
		break;
	case NOPBC:
		break;
	}
};

#define pbcCalc(r, box, x) r.x = r.x - floor(r.x / box.x)*box.x;

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif 
static void applyBoundaryCondition(VectorR &r, VectorR &box, PBCTYPE pbcType){
	switch (pbcType){
	case XYPBC:
		pbcCalc(r, box, x);
		pbcCalc(r, box, y);
		break;
	case XPBC:
		pbcCalc(r, box, x);
		break;
	case NOPBC:
		break;
	}
}

class ForceOnWalls { public: real x0, x1, y0, y1; };

inline real nint(real a){
	if (a >= 0.0) return floor(a + 0.5);
	else return floor(a - 0.5);
}


inline void a_dot_b(mat3x3 a, mat3x3 b, mat3x3 *c) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			c->m[i][j] = 0;
			for (int k = 0; k < 3; k++) {
				c->m[i][j] = c->m[i][j] + a.m[i][k] * b.m[k][j];
			}
		}
	}
}

//bool invertMatrix(const double m[16], double invOut[16])
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#endif 
static bool invertMatrix(real *m, real *invOut)
{
	real inv[9], det;
	int i;

	det = -m[2]*m[4]*m[6] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7] - m[0]*m[5]*m[7] - m[1]*m[3]*m[8] + m[0]*m[4]*m[8];
	if (det == 0)
		return false;

	inv[0] = -m[5] * m[7] + m[4] * m[8];
	inv[1] =  m[2] * m[7] - m[1] * m[8];
	inv[2] = -m[2] * m[4] + m[1] * m[5];
	inv[3] =  m[5] * m[6] - m[3] * m[8];
	inv[4] = -m[2] * m[6] + m[0] * m[8];
	inv[5] =  m[2] * m[3] - m[0] * m[5];
	inv[6] = -m[4] * m[6] + m[3] * m[7];
	inv[7] =  m[1] * m[6] - m[0] * m[7];
	inv[8] = -m[1] * m[3] + m[0] * m[4];

	det = 1.0f / det;

	for (i = 0; i < 9; i++)
		invOut[i] = inv[i] * det;

	return true;
}

void clearDevVector(dvector<int> &v);
void clearDevVector(dvector<real> &v);
void clearDevVector(dvector<VectorR> &v);
void clearDevVector(dvector<tetraIndexes> &v);
void clearDevVector(dvector<mat3x3> &v);

//template<typename T>
//void clearDevVector(dvector<T> &v);


#endif