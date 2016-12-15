
#include <vector>
#include "cuda_runtime.h"
#include "def.h"

inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

void md_device_init(int argc, char *argv[])
{
  int deviceCount;
  cudaError err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    fprintf(stderr, "error: cudaGetDeviceCount failed.\n");
    exit(EXIT_FAILURE);
  }
  if (deviceCount == 0) {
    fprintf(stderr, "error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  if (deviceCount == 1) {
    // just one device or an emulated device present, no choice
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
    if (err != cudaSuccess) {
      fprintf(stderr, "error: cudaGetDeviceProperties failed.\n");
      exit(EXIT_FAILURE);
    }
    if (deviceProp.major < 1) {
        fprintf(stderr, "error: device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
	md_device_report();
  }
  else {
    // several devices present, so make list of usable devices
    // and have one choosen among the currently available ones
    std::vector<int> usable_devices;
    for (int dev=0; dev<deviceCount; dev++) {
      cudaDeviceProp deviceProp;
      cudaError err = cudaGetDeviceProperties(&deviceProp, dev);
      if (err != cudaSuccess) {
        fprintf(stderr, "error: cudaGetDeviceProperties failed.\n");
        exit(EXIT_FAILURE);
      }
      if ((deviceProp.major >= 1) && (deviceProp.multiProcessorCount >= 2) && (deviceProp.computeMode != cudaComputeModeProhibited)) {
        usable_devices.push_back(dev);
      }
    }
    if (usable_devices.size() == 0) {
        fprintf(stderr, "error: no usable devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    //cudaError err = cudaSetValidDevices(&usable_devices[0], usable_devices.size());
    //if (err != cudaSuccess ) {
    //  fprintf(stderr, "error: cudaSetValidDevices failed.\n");
    //  exit(EXIT_FAILURE);
   // }
    // trigger device initialization by a non-device management function call
    cudaSetDevice(0);
    cudaThreadSynchronize();
    md_device_report();
  }
}

void md_device_report()
{

  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaDeviceProp devProp;
  cudaError err = cudaGetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) {
    fprintf(stderr, "error: cudaGetDeviceProperties failed.\n");
    exit(EXIT_FAILURE);
  }
  fprintf(stdout, "CUDA: Using device %d: %s\n", dev, prop.name);
  printf( " --- General Information for device %d ---\n", 0 );
  printf( "Name: %s\n", prop.name );
  printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
  printf( "Clock rate: %d\n", prop.clockRate );
  printf( "Device copy overlap: " );
  if (prop.deviceOverlap)
  printf( "Enabled\n" );
  else
  printf( "Disabled\n" );
  printf( "Kernel execition timeout : " );
  if (prop.kernelExecTimeoutEnabled)
  printf( "Enabled\n" );
  else
  printf( "Disabled\n" );
  printf( " --- Memory Information for device %d ---\n", 0 );
  printf( "Total global mem: %ld\n", prop.totalGlobalMem );
  printf( "Total constant Mem: %ld\n", prop.totalConstMem );
  printf( "Max mem pitch: %ld\n", prop.memPitch );
  printf( "Texture Alignment: %ld\n", prop.textureAlignment );
  return;
  
  /*
  devProp = prop;
  printf("#################################################\n");
  printf("Major revision number:         %d\n", devProp.major);
  printf("Minor revision number:         %d\n", devProp.minor);
  printf("Name:                          %s\n", devProp.name);
  printf("Total global memory:           %ld\n", devProp.totalGlobalMem);
  printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
  printf("Total shared memory per block: %ld\n", devProp.sharedMemPerBlock);
  printf("Total registers per block:     %ld\n", devProp.regsPerBlock);
  printf("Warp size:                     %d\n", devProp.warpSize);
  printf("Maximum memory pitch:          %u\n", devProp.memPitch);
  printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
	  printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
	  printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
  printf("Clock rate:                    %d\n", devProp.clockRate);
  printf("Total constant memory:         %u\n", devProp.totalConstMem);
  printf("Texture alignment:             %u\n", devProp.textureAlignment);
  printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
  printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
  */

  printf("################################################################\n");
	  int driverVersion = 0, runtimeVersion = 0;
	  //cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

  // Console log
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

  char msg[256];
  sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
	  (float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
  printf("%s", msg);

  printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
	  deviceProp.multiProcessorCount,
	  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
	  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
  printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
  // This is supported in CUDA 5.0 (runtime API device properties)
  printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize)
  {
	  printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
  }

#else
  // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
  int memoryClock;
  getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
  printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
  int memBusWidth;
  getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
  printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
  int L2CacheSize;
  getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

  if (L2CacheSize)
  {
	  printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
  }

#endif

  printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
	  deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
	  deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
	  deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
  printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
	  deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


  printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
  printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n", deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
  printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
	  deviceProp.maxThreadsDim[0],
	  deviceProp.maxThreadsDim[1],
	  deviceProp.maxThreadsDim[2]);
  printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
	  deviceProp.maxGridSize[0],
	  deviceProp.maxGridSize[1],
	  deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
  printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
  printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
  printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
  printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
  printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
  printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
  printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
  printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
  printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

  const char *sComputeMode[] =
  {
	  "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
	  "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
	  "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
	  "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
	  "Unknown",
	  NULL
  };
  printf("  Compute Mode:\n");
  printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

}

void resetDevce(){
	cudaDeviceReset();
}

void cudaCopyToDevice(void *d, void *h, int size){
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
}

class VecNormSqr{
public:
	__host__ __device__
		real operator()(const VectorR &a){
		return a.x*a.x + a.y*a.y;
	}
};

void squareVec(dvector<VectorR> &v, dvector<real> &vv){
	thrust::transform(v.begin(), v.end(), vv.begin(), VecNormSqr());
}

real findMax(dvector<real> &v){
	return thrust::reduce(v.begin(), v.end(), (real)(0.0), thrust::maximum<real>());
}

void multVec(dvector<real> &vv, dvector<real> &m ){
	//std::transform(a.begin(), a.end(), b.begin(), r.begin(), std::multiplies<float>());
	thrust::transform(vv.begin(), vv.end(), m.begin(), vv.begin(), thrust::multiplies<real>());
}

real __host__ reduce(dvector<real> &vv){
	return thrust::reduce(vv.begin(), vv.end());
}

/*
template<typename T>
void clearDevVector(dvector<T> &v){
	v.clear();
	thrust::device_vector<T>().swap(v);
}

template void clearDevVector(dvector<int> v);
*/

void clearDevVector(dvector<int> &v){
	v.clear();
	thrust::device_vector<int>().swap(v);
}

void clearDevVector(dvector < real > &v){
	v.clear();
	thrust::device_vector<real>().swap(v);
}

void clearDevVector(dvector < VectorR > &v){
	v.clear();
	thrust::device_vector<VectorR>().swap(v);
}

void clearDevVector(dvector < tetraIndexes > &v){
	v.clear();
	thrust::device_vector<tetraIndexes>().swap(v);
}

void clearDevVector(dvector < mat3x3 > &v){
	v.clear();
	thrust::device_vector<mat3x3>().swap(v);
}


