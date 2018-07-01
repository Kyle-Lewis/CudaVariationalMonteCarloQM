#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL_MEMBER __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL_MEMBER
#endif

#include <memory>
#include "GLInstance.h"

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "WritePNG.h"

// Because CUDA 8.0 doesn't have it
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};


class VMCSimulation
{
	public:
		VMCSimulation(const char* configFileName);
		void LoadSettings(const char* configFileName);
		void InitThreadDimensions();
		void InitPaths();
		void RunSimulation();
		~VMCSimulation();

		enum struct Mode : int
		{
			HarmonicOscillator = 0,
			DoubleWell = 1
		};

		Mode StringToPotentialMode(std::string input);

		// Settings to be loaded from config file
		struct Settings
		{
			unsigned int simWidth;      // determines bin count for histograms
			unsigned int simHeight;		// determines bin count for histograms
  			unsigned int numPoints;		// number of points per ring
			float xRange;				// spatial range: +-xRange/2
			float yRange;				// spatial range: +-yRange/2
			float epsilon;				// Maximum Monte Carlo variation in either X or Y
			float tau;					// Total Time along a path
			float dt;					// Time discretization, tau/numPoints
			Mode mode;					// Determines the form of the potential for the simulation
			
			bool recording;
			unsigned int frames;		// When recording, how many frames to record

		};

		/// Derived Settings ///
		float dt;						// Time discretization along path
		int xBlocks;
		int threadsPerBlock;
		dim3 tpbColor;
		dim3 colorBlocks;

		unsigned int redBlack;

		Settings settings;							// Settings instance 

		thrust::host_vector<float2> ringPoints;		// Host data ignored after initialization
		thrust::device_vector<float2> dRingPoints;	// Device data (necessary?)
		float2* rawRingPoints;						// Actual device data to operate on

		thrust::host_vector<int> binSums;
		thrust::device_vector<int> dBinSums;
		int* rawBinSums;

		thrust::host_vector<float3> colorMap;
		thrust::device_vector<float3> dColorMap;
		float3* rawColorMap;

		thrust::host_vector<unsigned char> pixelData;
		thrust::device_vector<unsigned char> dPixelData;
		unsigned char* rawPixelData;

		std::unique_ptr<GLInstance> glInstance;
};


CUDA_KERNEL_MEMBER void Step 			 (float2* rawRingPoints,
										  float epsilon,
										  int mode,
										  float tau, 
										  float dt, 
										  int numPoints,
										  unsigned int redBlack);

CUDA_KERNEL_MEMBER void Zero_Histogram 	 (int* rawBinSums,
									      unsigned int simHeight,
									      unsigned int simWidth);

CUDA_KERNEL_MEMBER void PopulateBins	 (int* rawBinSums,
										  float2* rawRingPoints,
										  float xRange,
										  float yRange,
										  int mode, 
										  int numPoints,
										  unsigned int simHeight,
										  unsigned int simWidth);

CUDA_KERNEL_MEMBER void ExpectationEnergy(float2* rawRingPoints,
										  float* energy,
										  float dt,
										  int mode,
										  int numPoints);

CUDA_KERNEL_MEMBER void Color 			 (float3* colors,
						  				  float3* colorMap,
						  				  int* rawBinSums,
									  	  int mapMin,
										  int mapMax,
										  unsigned int simWidth,
										  unsigned int simHeight);

CUDA_CALLABLE_MEMBER float Action		 (float2* rawRingPoints,
										  float2 newPoint,
										  int mode,
										  float dt,
										  int numPoints,
									      unsigned int idx);

CUDA_KERNEL_MEMBER void FormPNGData 	 (float3* colors,
										  unsigned char* pixelData, 
										  unsigned int simWidth, 
										  unsigned int simHeight);

CUDA_CALLABLE_MEMBER float Potential 	 (float2 point,
					   					  int mode);
