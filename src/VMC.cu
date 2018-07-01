#include "VMC.h"
//#include "Step.cu"
//#include "ExpectationEnergy.cu"


VMCSimulation::VMCSimulation(const char* config)
{
	LoadSettings(config);
	std::cout<<"	Created VMC Simulation object"<<std::endl;
	InitThreadDimensions();
	glInstance = make_unique<GLInstance>(settings.simWidth, settings.simHeight);
	std::cout<<"	Created openGL Instance for the Simulation"<<std::endl;
	InitPaths();
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
		else if (line.find("recording") != std::string::npos)
		{
			settings.recording = bool(stoi(line.substr(9)));
			std::cout<<"	Recording: " << settings.recording << std::endl;
		}
		else if (line.find("frames") != std::string::npos)
		{
			settings.frames = stoi(line.substr(6));
			std::cout<<"	Frames: " << settings.frames << std::endl;
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
		//if (i <settings.numPoints / 2)
			ringPoints[i] = make_float2(0.1, 0.1);
		//else
	//		ringPoints[i] = make_float2(-0.01, 0.0);

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
	colorfile.close();

	dColorMap = colorMap;
	rawColorMap = thrust::raw_pointer_cast(dColorMap.data());

		// set pixel count (3x for 3x8 bit color format) and establish 
		// those vectors on host and device 
	pixelData.resize(3 * settings.simWidth * settings.simHeight);
	dPixelData = pixelData; 
	rawPixelData = thrust::raw_pointer_cast(dPixelData.data());

	cudaThreadSynchronize();
}

void VMCSimulation::RunSimulation()
{
	// Info stuff
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float fpsTime = 0.0;
	int steps = 0;

	int mapMin = 0;
	int mapMax = 0;
	thrust::pair<thrust::device_ptr<int>, thrust::device_ptr<int>> minMaxPtrs;

	// Temporary, don't need these
	float* hAvgEnergy;
	float* dAvgEnergy;
	checkCuda( cudaMalloc((void**)&dAvgEnergy, sizeof(float)));
	checkCuda( cudaMallocHost((void**)&hAvgEnergy, sizeof(float))); 
	hAvgEnergy[0] = 0.0;
	checkCuda( cudaMemcpy(dAvgEnergy, hAvgEnergy, sizeof(float), cudaMemcpyHostToDevice));
	float* zerofloat;
	checkCuda( cudaMallocHost((void**)&zerofloat, sizeof(float)));
	zerofloat[0] = 0.0;

	std::string frameName;

	while(!glfwWindowShouldClose(glInstance->window))
	{
		cudaEventRecord(start, 0);
		
		float3 *devPtr;
		checkCuda(cudaGraphicsMapResources(1, &glInstance->cudaColorResource, 0));

		size_t numBytes;
		checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes,
													   *&glInstance->cudaColorResource));
		
		//std::cout<<numBytes<<std::endl;

		Step<<< xBlocks, threadsPerBlock >>>				(rawRingPoints,
														     settings.epsilon,
														     static_cast<int>(settings.mode),
														     settings.tau, 
														     settings.dt, 
														     settings.numPoints,
														     redBlack);
		gpuErrchk(cudaPeekAtLastError());
		checkCuda(cudaDeviceSynchronize());

		ExpectationEnergy<<< xBlocks, threadsPerBlock >>> 	(rawRingPoints,
														     dAvgEnergy,
														     settings.dt,
														     static_cast<int>(settings.mode),
														     settings.numPoints);
		gpuErrchk(cudaPeekAtLastError());
		checkCuda(cudaDeviceSynchronize());

		checkCuda( cudaMemcpy(hAvgEnergy, dAvgEnergy, sizeof(float), cudaMemcpyDeviceToHost));
		checkCuda( cudaMemcpy(dAvgEnergy, zerofloat, sizeof(float), cudaMemcpyHostToDevice));

	    PopulateBins<<< xBlocks, threadsPerBlock >>>		(rawBinSums,
				    										 rawRingPoints,
				    										 settings.xRange,
				    										 settings.yRange,
												     	     static_cast<int>(settings.mode),
				    										 settings.numPoints,
				    										 settings.simHeight,
				    										 settings.simWidth);
		gpuErrchk(cudaPeekAtLastError());

		checkCuda(cudaDeviceSynchronize());
		minMaxPtrs = thrust::minmax_element(thrust::device, thrust::device_pointer_cast(rawBinSums), thrust::device_pointer_cast(rawBinSums) + (settings.simHeight * settings.simWidth) - 1 );
		mapMin = *minMaxPtrs.first;
		mapMax = *minMaxPtrs.second;
		Color<<< colorBlocks, tpbColor >>> 					(devPtr,
															 rawColorMap,
														  	 rawBinSums,
														 	 mapMin,
															 mapMax,
															 settings.simWidth, 
															 settings.simHeight);
		gpuErrchk(cudaPeekAtLastError());
		Zero_Histogram<<< colorBlocks, tpbColor >>>			(rawBinSums,
												    		 settings.simWidth,
												    		 settings.simHeight);
	    gpuErrchk(cudaPeekAtLastError());
		checkCuda(cudaDeviceSynchronize());

		(redBlack == 0) ? redBlack = 1 : redBlack = 0;

		glInstance->Draw();

		if (settings.recording && steps < settings.frames)
		{
			frameName = FrameNameGen(steps, settings.frames);
			FormPNGData<<< colorBlocks, tpbColor >>> 		(devPtr, 
															 rawPixelData, 
															 settings.simWidth, 
															 settings.simHeight);

			gpuErrchk(cudaPeekAtLastError());

			cudaMemcpy(pixelData.data(), rawPixelData, settings.simWidth * settings.simHeight * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			gpuErrchk(cudaPeekAtLastError());
			WritePNG(pixelData.data(), frameName, settings.simWidth, settings.simHeight);
		}

		steps++;
		//cudaEventRecord(stop, 0);
		//cudaEventElapsedTime(&fpsTime, start, stop); 
		char title[512];
		sprintf(title, "Cuda Variational Monte Carlo: %12.2f fps, path count: %u, steps taken: %d", 1.0f/(fpsTime/1000.0f), settings.numPoints, steps);
		glfwSetWindowTitle(glInstance->window, title);

		checkCuda(cudaGraphicsUnmapResources(1, &glInstance->cudaColorResource, 0));

		if(glfwGetKey(glInstance->window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(glInstance->window, 1);
			std::cout << "Window Close Set, next loop should end" << std::endl;
			std::cout <<" wow " << std::endl;
		}
	}
	std::cout << "Exiting simulation function" << std::endl;
	//free(data);
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

	switch(settings.simWidth * settings.simHeight)
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
	}
			
	threadsPerBlock = settings.numPoints/xBlocks;
	std::cout<<"	Calling path algorithm kernels with:"<<std::endl
			 <<"	ThreadsPerBlock: ["<<threadsPerBlock<<"]"<<std::endl
			 <<"	On a Grid of: ["<<xBlocks<<"] Blocks"<<std::endl<<std::endl;
    std::cout<<"	Calling painting kernels with:"<<std::endl
    		 <<"	ThreadsPerBlock: ["<<tpbColor.x<<","<<tpbColor.y<<"]"<<std::endl
    		 <<"	On a Grid of: ["<<colorBlocks.x<<","<<colorBlocks.y<<"]"<<std::endl;
}

VMCSimulation::~VMCSimulation()
{
}
