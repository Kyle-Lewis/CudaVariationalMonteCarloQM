#include "VMC.h"

		 /* measure the action change and roll for a result
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
	if (idx >= numPoints)
		printf("ERROR: Out of rawRingPoints[] bounds, calling: %d, max: %d\n", idx, numPoints);
	
	// alternate steps, red/black:
	if((idx + redBlack) % 2 == 0)
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

	// Run accept/deny on the move
	rawRingPoints[idx] = MCTest(rawRingPoints, newPoint, mode, tau, dt, numPoints, idx);
	__syncthreads();
}