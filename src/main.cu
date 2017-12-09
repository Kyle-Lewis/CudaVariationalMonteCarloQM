#include "VMC.h"

using namespace std;

int main(int argc, char *argv[])
{
	VMCSimulation simulation {"config"};

	cout<< "hello" << endl;
	cout<<"xBlocks after init: "<<simulation.xBlocks<<endl;
		// Getting stuff that I want from the glInstance as local pointers,
		// Not sure if this is a good move but I don't like simulation.glInstance->window vs "window"
	//GLFWwindow* window = simulation.glInstance->window;

	//simulation.RunSimulation();
	
	return 0;
}