#include "VMC.h"

VMCSimulation::VMCSimulation(const char* config)
{
	LoadSettings(config);
	std::cout<<"	Created VMC Simulation object"<<std::endl;
	glInstance = make_unique<GLInstance>(settings.simWidth, settings.simHeight);
	std::cout<<"	Created openGL Instance for the Simulation"<<std::endl;
	InitThreadDimensions();
	InitPaths();
	//std::cout<<"xBLokcs here: "<< xBlocks<<std::endl;
	redBlack = 0; 
}

VMCSimulation::Mode VMCSimulation::StringToPotentialMode(std::string input)
{
	std::cout << "	Input: " << input << std::endl;
	if (input == " HarmonicOscillator"){return Mode::HarmonicOscillator;}
	if (input == " DoubleWell"){return Mode::DoubleWell;}
	std::cout << "	Didn't recognize config mode as a simulation mode, using default: HarmonicOscillator"<<std::endl;
	return Mode::HarmonicOscillator;			
}

void VMCSimulation::LoadSettings(const char* configFileName)
{
	using namespace std;
	ifstream configFile;
	configFile.open(configFileName);
	string line;
	std::cout<<"	Using Settings:"<<std::endl;
	while (getline(configFile, line))
	{
		if (line.find("simWidth") != std::string::npos) 
		{ 
			settings.simWidth = stoi(line.substr(8));
			std::cout<<"	simWidth: "<<settings.simWidth<<endl;
		}
		else if (line.find("simHeight") != std::string::npos) 
		{ 
			settings.simHeight = stoi(line.substr(9));
			std::cout<<"	simHeight: "<<settings.simHeight<<std::endl;
		}
		else if (line.find("numPoints") != std::string::npos)
		{ 
			settings.numPoints = stoi(line.substr(9)); 
			std::cout<<"	numPoints: "<<settings.numPoints<<std::endl; 
		}
		else if (line.find("xRange") != std::string::npos) 
		{ 
			settings.xRange = stof(line.substr(6));
			std::cout<<"	xRange: "<<settings.xRange<<std::endl;
		}
		else if (line.find("yRange") != std::string::npos) 
		{ 
			settings.yRange = stof(line.substr(6)); 
			std::cout<<"	yRange: "<<settings.yRange<<std::endl;
		}
		else if (line.find("epsilon") != std::string::npos)
		{	
			settings.epsilon = stof(line.substr(7));
			std::cout<<"	epsilon: "<<settings.epsilon<<std::endl;
		}
		else if (line.find("tau") != std::string::npos)
		{
			settings.tau = stof(line.substr(3));
			std::cout<<"	tau: "<<settings.tau<<std::endl;
		}
		else if (line.find("mode") != std::string::npos) 
		{ 
			settings.mode = StringToPotentialMode(line.substr(4));
			std::cout<<"	Potential Mode: "<<line.substr(4)<<std::endl;
		}
	}
	settings.dt = settings.tau/settings.numPoints;
	std::cout<<"	dt:"<<settings.dt<<std::endl;

	std::cout<<std::endl<<"	Press Enter to continue"<<std::endl;
	configFile.close();
	cin.ignore();
}

void VMCSimulation::InitPaths()
{
	ringPoints.resize(settings.numPoints);
	for (int i = 0; i < settings.numPoints; i++)
	{
		if (i < settings.numPoints/2)
			ringPoints[i] = make_float2(0.5, 0.0);
		else
			ringPoints[i] = make_float2(0.5, 0.0);
	}
	dRingPoints = ringPoints;
	rawRingPoints = thrust::raw_pointer_cast(dRingPoints.data());

	binSums.resize(settings.simWidth * settings.simHeight);
	for (int i = 0; i < settings.simWidth * settings.simHeight; i++)
		binSums[i] = 0;
	dBinSums = binSums;
	rawBinSums = thrust::raw_pointer_cast(dBinSums.data());

	colorMap.resize(512);
	std::ifstream colorfile("data/Hot_Cold_No_Zero", std::ifstream::in);
	std::string colorLine;
	int i = 0;
	while(getline(colorfile, colorLine)){
		std::stringstream linestream(colorLine);
		linestream >> colorMap[i].x >> colorMap[i].y >> colorMap[i].z;
		i++;
	}
	dColorMap = colorMap;
	rawColorMap = thrust::raw_pointer_cast(dColorMap.data());

	cudaThreadSynchronize();
}

void VMCSimulation::RunSimulation()
{

	// Info stuff
	//cudaEvent_t start, stop;
	float fpsTime = 0.0;
	int steps = 0;

	std::cout<<xBlocks<<" "<<threadsPerBlock<<std::endl;
	//std::cout<<std::endl<<"	Press Enter to continue"<<std::endl;
    
	// TODO its own initialization for colorization
	dim3 tpbColor;
	dim3 colorBlocks;
	if (settings.simWidth * settings.simHeight == 262144)
	{
		int xColorBlocks = 2;
		int yColorBlocks = 256;
		tpbColor.x = settings.simWidth/xColorBlocks;
		tpbColor.y = settings.simHeight/yColorBlocks;
		colorBlocks.x = xColorBlocks;
		colorBlocks.y = yColorBlocks;
	}
	else if (settings.simWidth * settings.simHeight == 16384)
	{
		int xColorBlocks = 1;
		int yColorBlocks = 128;
		tpbColor.x = settings.simWidth/xColorBlocks;
		tpbColor.y = settings.simHeight/yColorBlocks;
		colorBlocks.x = xColorBlocks;
		colorBlocks.y = yColorBlocks;
	}
	else if (settings.simWidth * settings.simHeight == 65536)
	{
		int xColorBlocks = 1;
		int yColorBlocks = 256;
		tpbColor.x = settings.simWidth/xColorBlocks;
		tpbColor.y = settings.simHeight/yColorBlocks;
		colorBlocks.x = xColorBlocks;
		colorBlocks.y = yColorBlocks;
	}
	std::cout<<"	Calling painting kernels with:"<<std::endl
		 <<"	ThreadsPerBlock: ["<<tpbColor.x<<","<<tpbColor.y<<"]"<<std::endl
		 <<"	On a Grid of: ["<<colorBlocks.x<<","<<colorBlocks.y<<"]"<<std::endl;
	//std::cout<<"colorBlocks: "<<colorBlocks.x<<" "<<colorBlocks.y<<std::endl;
	//std::cout<<"tbpColor: "<<tpbColor.x<<" "<<tpbColor.y<<std::endl;
	int mapMin = 0;
	int mapMax = 0;
	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> minMaxPtrs;

	while(!glfwWindowShouldClose(glInstance->window) && steps < 10)
	{
		//cudaEventRecord(start, 0);
		//std::cout<<"happens "<<steps<<std::endl;
		
		float3 *devPtr;
		checkCuda(cudaGraphicsMapResources(1, &glInstance->cudaColorResource, 0));

		size_t numBytes;
		checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
													   *&glInstance->cudaColorResource));
		
		/*std::cout<<numBytes<<std::endl;

		Step<<< xBlocks, threadsPerBlock >>>(rawRingPoints,
										     settings.epsilon,
										     static_cast<int>(settings.mode),
										     settings.tau, 
										     settings.dt, 
										     settings.numPoints,
										     redBlack);
		checkCuda(cudaDeviceSynchronize());
		
	    PopulateBins<<< xBlocks, threadsPerBlock >>>(rawBinSums,
		    										 rawRingPoints,
		    										 settings.xRange,
		    										 settings.yRange,
										     	     static_cast<int>(settings.mode),
		    										 settings.numPoints,
		    										 settings.simHeight,
		    										 settings.simWidth);
		    										 
		checkCuda(cudaDeviceSynchronize());
		minMaxPtrs = thrust::minmax_element(thrust::device, thrust::device_pointer_cast(rawBinSums), thrust::device_pointer_cast(rawBinSums) + (settings.simHeight * settings.simWidth) - 1 );
		mapMin = *minMaxPtrs.first;
		mapMax = *minMaxPtrs.second;
		std::cout <<mapMin<<" "<<mapMax<<std::endl;
		Color<<< colorBlocks, tpbColor >>> (devPtr,
											rawColorMap,
											rawBinSums,
											mapMin,
											mapMax,
											settings.simWidth, 
											settings.simHeight);

		//std::cout<<"After call devPtr[10]: "<<devPtr[10].x<<" "<<devPtr[10].y<<" "<<devPtr[10].z<<std::endl;
		Zero_Histogram<<< colorBlocks, tpbColor >>>(rawBinSums,
												    settings.simWidth,
												    settings.simHeight);

		checkCuda(cudaDeviceSynchronize());

		(redBlack == 0) ? redBlack = 1 : redBlack = 0; // flips: 0,1 : even,odd*/

		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//		glUseProgram(glInstance->shaderProgram);
//		glBindVertexArray(glInstance->vertexArray);
//
	//	glDrawArrays(GL_QUADS, 0, glInstance->colorIndexCount);
//
	//	glfwSwapBuffers(glInstance->window);
//		glfwPollEvents();
		glInstance->Draw();

		// Title and infonumPoints
		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&fpsTime, start, stop);
		steps++;
		char title[256];
		sprintf(title, "Cuda Variational Monte Carlo: %12.2f fps, path count: %u, steps taken: %d", 1.0f/(fpsTime/1000.0f), settings.numPoints, steps);
		glfwSetWindowTitle(glInstance->window, title);

		checkCuda(cudaGraphicsUnmapResources(1, &glInstance->cudaColorResource, 0));

		if(glfwGetKey(glInstance->window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(glInstance->window, 1);
		}
	}
	std::cout<<mapMin<<std::endl;
	std::cout<<mapMax<<std::endl;
}

void VMCSimulation::InitThreadDimensions()
{
	switch(settings.numPoints){
		case 256:
			xBlocks = 16;
			break;
		case 1024:
			xBlocks = 4;
			break;
		case 4096:
			xBlocks = 16;
			break;
		case 16384:
			xBlocks = 64;
			break;
		case 32768:
			xBlocks = 128;
			break;
		case 65536:
			xBlocks = 256;
			break;
		case 262144:
			xBlocks = 1024; // max
			break;
		default:
			std::cout<<"Bad Dimensions"<<std::endl;
			exit(1);
	}

	/*switch(settings.simWidth * settings.simHeight)
	{
		// 128 x 128
		case 16384:
			tpbColor.x = settings.simWidth/1;
			tpbColor.y = settings.simHeight/128;
			colorBlocks.x = 1;
			colorBlocks.y = 128;
			break;
		// 256 x 256
		case 65536:
			tpbColor.x = settings.simWidth/1;
			tpbColor.y = settings.simHeight/256;
			colorBlocks.x = 1;
			colorBlocks.y = 256;
			break;
		// 512 x 512
		case 262144:
			tpbColor.x = settings.simWidth/2;
			tpbColor.y = settings.simHeight/256;
			colorBlocks.x = 2;
			colorBlocks.y = 256;
			break;
		default:
			std::cout<<"Bad Dimensions"<<std::endl;
			exit(1);
	}*/
			
	threadsPerBlock = settings.numPoints/xBlocks;
	std::cout<<"	Calling path algorithm kernels with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<threadsPerBlock<<"]"<<std::endl
			 <<"	On a Grid of: ["<<xBlocks<<"] Blocks"<<std::endl<<std::endl;
    /*std::cout<<"	Calling painting kernels with:"<<std::endl
    		 <<"	ThreadsPerBlock: ["<<tpbColor.x<<","<<tpbColor.y<<"]"<<std::endl
    		 <<"	On a Grid of: ["<<colorBlocks.x<<","<<colorBlocks.y<<"]"<<std::endl;*/
}

VMCSimulation::~VMCSimulation()
{

}

															////////////////////////
															//**** Cuda Calls ****//
															////////////////////////

__global__ void Zero_Histogram(int* rawBinSums,
							   unsigned int simHeight,
							   unsigned int simWidth)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	rawBinSums[x + y * simWidth] = 0;
}

__global__ void PopulateBins(int* rawBinSums,
							 float2* rawRingPoints,
							 float xRange,
							 float yRange,
							 int mode, 
							 int numPoints,
							 unsigned int simHeight,
							 unsigned int simWidth)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int xBin = 0;
	unsigned int yBin = 0;

	while (rawRingPoints[idx].x > (-xRange / 2.0 + xBin * xRange / simWidth)) xBin++;
	while (rawRingPoints[idx].y > (-yRange / 2.0 + yBin * yRange / simHeight)) yBin++;

	/*if (idx == 10)
	{
		printf("Idx 10 reads X: %f, Y: %f\n", rawRingPoints[idx].x, rawRingPoints[idx].y);
		printf("Idx 10 trying to add to bin: %d, %d, %d\n", xBin, yBin, xBin + yBin * simWidth);
	}*/

	if(xBin > simWidth - 1 || yBin > simHeight - 1){return;};

	atomicAdd(&rawBinSums[xBin + yBin * simWidth], 1);
}

__global__ void Color(float3* colors,
					  float3* rawColorMap,
					  int* rawBinSums,
					  int mapMin,
					  int mapMax,
					  unsigned int simWidth,
					  unsigned int simHeight) // TODO
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int quadIdx = x + simWidth*y;
	if (y < 12)
	{
		printf("QuadIdx: %d SimWidth: %d X: %d, Y: %d\n", quadIdx, simWidth, x, y);
	}
	/*
	if(quadIdx == 20)
	{
		printf("colors from last time at idx 20: %f, %f, %f\n", colors[4*quadIdx].x, colors[4*quadIdx].y, colors[4*quadIdx].z);
		printf("idx 20 position, expected color:%d, y:%d, with %f, %f, %f\n", x, y, rawColorMap[150].x, rawColorMap[150].y, rawColorMap[150].z);
	}*/
	int mapped = (int)(0 + (((rawBinSums[quadIdx] - mapMin) * (511 - 0)) / (mapMax - mapMin)));
	/*if((x+1)*(y+1) >= simWidth * simHeight)
	{
		printf("HAPPENED, %d, %d\n", x, y);
		printf("color sample: %f, %f, %f, \n", rawColorMap[0].x, rawColorMap[0].y, rawColorMap[0].z);
		printf("mapped value: %d\n", mapped);
	}*/
	for(int i = 0; i < 4; i++)
	{
		if(rawBinSums[quadIdx] >= 511 || rawBinSums[quadIdx] <= 0)
		{
			colors[4*quadIdx+i] = rawColorMap[511];
		}
		else
		{
			colors[4*quadIdx+i] = rawColorMap[mapped];
			if(quadIdx == 200)
			{
				printf("color sample: %f, %f, %f, \n", rawColorMap[mapped].x, rawColorMap[mapped].y, rawColorMap[mapped].z);
			}
		}
	}
}
		/* Returns the potential at a given x,y coordinate
		 * depending on the potential mode of the simulation */

__device__ float Potential(float2 point,
						   int mode)
{
	if (mode == 0)
	{
		//return 0.5 * r^2
		return 0.5 * (point.x * point.x + point.y * point.y);
	}
	else if (mode == 1)
	{
		// TODO 
		float alpha = 0.2;
		float beta = 2.0;
		return alpha*pow(point.x,4.0) - beta*pow(point.x,2.0) +
			  pow(beta,2.0)/(4.0*alpha) + pow(point.y,2.0);
	}

	// unrecognized mode, shouldn't ever happen
	return 0.0;
}

		/* Returns the action associated with a particular 
		 * slice of the discrete path. Using the imaginary 
		 * time black magic we add, instead of subtract the 
		 * potential contribution: V */

__device__ float Action(float2* rawRingPoints,
						float2 newPoint,						// the new point on the path generated by the random move
						int mode,
						float dt,
						int numPoints,
					    unsigned int idx)								// the idx of the point that changed
{
	float2 K; // vector components of momentum

	if (idx == 0) // Point is at the beginning of the ring
	{
		K.x = 0.5 * (newPoint.x - rawRingPoints[idx+1].x) * (rawRingPoints[numPoints].x - newPoint.x) / dt;
		K.y = 0.5 * (newPoint.y - rawRingPoints[idx+1].y) * (rawRingPoints[numPoints].y - newPoint.y) / dt;
	}
	else if (idx == numPoints) // Point is at the end of the ring
	{
		K.x = 0.5 * (newPoint.x - rawRingPoints[idx+1].x) * (rawRingPoints[0].x - newPoint.x) / dt;
		K.y = 0.5 * (newPoint.y - rawRingPoints[idx+1].y) * (rawRingPoints[0].y - newPoint.y) / dt;
	}
	else 
	{
		K.x = 0.5 * (newPoint.x - rawRingPoints[idx-1].x) * (rawRingPoints[idx+1].x - newPoint.x) / dt;
		K.y = 0.5 * (newPoint.y - rawRingPoints[idx-1].y) * (rawRingPoints[idx+1].y - newPoint.y) / dt;
	}

	float magK = pow((K.x * K.x + K.y * K.y), 0.5); 
	float V = Potential(newPoint, mode);
	return magK + V;
}

		/* A Monte Carlo step for any variable of interest,
		 * measure the action change and roll for a result
		 * if necessary. This metd just compacts the repeated uses
		 * within Step() */

__device__ float2 MCTest(float2* rawRingPoints,
						 float2 newPoint,
						 int mode,
						 float tau,
						 float dt,
						 int numPoints,
						 unsigned int thdIdx)
{
	curandState state;
	curand_init((unsigned long long)clock(), thdIdx, 0, &state);
	float S_old = Action(rawRingPoints, rawRingPoints[thdIdx], mode, dt, numPoints, thdIdx);
	float S_new = Action(rawRingPoints, newPoint, mode, dt, numPoints, thdIdx);
		// If action is decreased by the path always accept 
	if(S_new < S_old)
	{
		return newPoint;
	}
		// If action is increased accept w/ probability 
	else
	{
		float roll = curand_uniform(&state);
		float prob = expf(-S_new/(1.0/tau)) / expf(-S_old/(1.0/tau));
		if(roll < prob){
			return newPoint;
		}
	}	
	return rawRingPoints[thdIdx]; // simply return the existing point
}

		/* Steps alternating even/odd "red/black" points in the path
		 * with the chance of acceptance being related to the change 
		 * in the action which the move brings to the path */
		// Note: Kernel Functions can't be defined within the class

__global__ void Step(float2* rawRingPoints,
					 float epsilon,
					 int mode,
					 float tau, 
					 float dt, 
					 int numPoints,
					 unsigned int redBlack)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

		// alternate steps, red/black:
	if((idx + redBlack)%2 == 0)
	{
		return;
	}

	float2 newPoint = rawRingPoints[idx];

	curandState state;
	curand_init((unsigned long long)clock(), idx, 0, &state);

	float randX = (0.5 - curand_uniform(&state)) * 2.0 * epsilon;
	float randY = (0.5 - curand_uniform(&state)) * 2.0 * epsilon;
	newPoint.x = rawRingPoints[idx].x + randX;
	newPoint.y = rawRingPoints[idx].y + randY;
	/*if(idx == 15)
	{
		printf("16th point: %f, %f, randX: %f, randY: %f\n", rawRingPoints[idx].x,rawRingPoints[idx].y, randX, randY);
	}*/
		// Run accept/deny on the move
	rawRingPoints[idx] = MCTest(rawRingPoints, newPoint, mode, tau, dt, numPoints, idx);
	__syncthreads();
}

