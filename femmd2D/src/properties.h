#ifndef __PROPS_H__
#define __PROPS_H__

#include "def.h"

struct Prop{
	real val, sum, sum2, avgval, stdv, lastAvg;
	Prop(){ val = 0; sum = 0; sum2 = 0; /*avgval = 0; stdv = 0;*/ }
	void zero(){ val = 0; sum = 0; sum2 = 0; }
	void accum(){ sum += val; sum2 += val*val; }
	void avg(int n){ avgval = sum / n; stdv = sqrt(Max((sum2 / n - avgval*avgval), 0.0)); }
	void print(){ printf(" prop: val=%f sum=%f sum2=%f avgval=%f stdv=%f ", val, sum, sum2, avgval, stdv); }
};

class MDSystem;

class Properties{
public:
	MDSystem *sys;
	int step_avg;
	int step_equi;
	int count;
	Properties(){
		step_avg = 1000; 
		count = 0;
		step_equi = 10000;
		countRdf = 0;
		skipStepsRdf = 10;
		sizeHistRdf = 100;
		rangeRdf = 5;
		avgCountRdf = 30000;
	}
	Prop temperature;
	Prop uPotential;
	Prop uKinetic;
	Prop clusterUKinetic;
	Prop uTotal;
	Prop uBonded;
	Prop uFea;
	Prop pressure;
	Prop molPressure;
	Prop virial;
	Prop feaVirial;
	Prop clusterVirial;

	void evaluate();
	void compute();
	void accumulate(int icode);
	void zerovals(){
		temperature.val = uKinetic.val = 0;
		uPotential.val = uBonded.val = uFea.val = virial.val = pressure.val = 0;
		clusterUKinetic.val = feaVirial.val = clusterVirial.val = 0;
		virial.val = 0;
		molPressure.val = 0;
	}
	void zero(){
		count = 0;
		uPotential.zero(); uKinetic.zero(); virial.zero(); pressure.zero();
		temperature.zero(); uBonded.zero(); uFea.zero();
		clusterUKinetic.zero(); feaVirial.zero(); clusterVirial.zero();
		molPressure.zero();
	}

	int countRdf;
	int skipStepsRdf;
	int sizeHistRdf;
	real histRdf[500];
	real rangeRdf;
	int avgCountRdf;
	void evalRdf();
	void evalClusterRdf();
	void reset() { count = 0; countRdf = 0; evalClusterRdf(); }
};

#endif
