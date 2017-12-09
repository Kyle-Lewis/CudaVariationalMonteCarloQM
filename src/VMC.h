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

// Because CUDA 8.0 doesn't have it
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
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

		// Kernal Calls:
		//CUDA_KERNEL_MEMBER void Step();
		// Device Calls:
		//CUDA_CALLABLE_MEMBER static float Action(float2 point, int idx);
		//CUDA_CALLABLE_MEMBER static float2 MCTest(float2 newPoint, unsigned int thdIdx);
		//CUDA_CALLABLE_MEMBER static float Potential(float2 point);

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


		std::unique_ptr<GLInstance> glInstance;
	private:
		const char *vertexShaderText = 
		"#version 450 \n"
		"layout(location = 0) in vec3 vertexPosition;"
		"layout(location = 1) in vec3 vertexColor;"
		"out vec3 color;"
		"void main() {"
		"	color = vertexColor;"
		"	gl_Position = vec4(vertexPosition, 1.0);"
		"}";

		const char *fragmentShaderText = 
		"#version 450\n"
		"in vec3 color;"
		"out vec4 fragmentColor;"
		"void main() {"
		"	fragmentColor = vec4(color, 1.0);"
		"}";
};


CUDA_KERNEL_MEMBER void Step(float2* rawRingPoints,
							 float epsilon,
							 int mode,
							 float tau, 
							 float dt, 
							 int numPoints,
							 unsigned int redBlack);

CUDA_KERNEL_MEMBER void Zero_Histogram(int* rawBinSums,
									   unsigned int simHeight,
									   unsigned int simWidth);

CUDA_KERNEL_MEMBER void PopulateBins(int* rawBinSums,
									 float2* rawRingPoints,
									 float xRange,
									 float yRange,
									 int mode, 
									 int numPoints,
									 unsigned int simHeight,
									 unsigned int simWidth);

CUDA_KERNEL_MEMBER void Color(float3* colors,
							  float3* colorMap,
							  int* rawBinSums,
							  int mapMin,
							  int mapMax,
							  unsigned int simWidth,
							  unsigned int simHeight);