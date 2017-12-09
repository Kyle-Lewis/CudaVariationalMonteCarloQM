#include "GLInstance.h"

GLInstance::GLInstance(int simWidth, int simHeight)
{

		// Init and generate window of proper size

	if (!glfwInit()) 
	{
		fprintf(stderr, "ERROR: could not start GLFW3\n");
	}
	std::cout<<"happened1"<<std::endl;
	window = glfwCreateWindow(512, 512, "Cuda Fluid Dynamics", NULL, NULL);
	glfwSetWindowPos(window, 1920 - 2*simWidth - 2, 0);
	std::cout<<"happened2"<<std::endl;

	if (!window) 
	{
		fprintf(stderr, "ERROR: could not open window with GLFW3\n");
		glfwTerminate();
	}
	std::cout<<"happened3"<<std::endl;

	glfwMakeContextCurrent(window);	
	glewExperimental = GL_TRUE;
	glewInit();
	std::cout<<"happened4"<<std::endl;

		// Init and generate point and color data

	colorIndexCount = simWidth * simHeight * 4;
	quadPoints = (float3*)malloc(colorIndexCount*sizeof(float3));
	colors = (float3*)malloc(colorIndexCount*sizeof(float3));

	int idxRow = 0;
	int idxCol = 0;
	float halfHeight = 0.0 ; 
	float halfWidth = 0.0 ; 
	for (int i = 0; i < colorIndexCount; i++){
		if(i%4==0){ //top left of a quad
			quadPoints[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==1){ //bottom left of a quad
			quadPoints[i].x = 2.0*(float)idxCol/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfWidth;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==2){ //bottom right of a quad
			quadPoints[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)(idxRow+1)/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
		}
		if(i%4==3){ //top right of a quad
			quadPoints[i].x = 2.0*(float)(idxCol+1)/simWidth - 1.0f + halfWidth;
			quadPoints[i].y = 1.0f - 2.0*(float)idxRow/simHeight - halfHeight;
			quadPoints[i].z = 0.0f;
			idxCol++;
		}
		if(idxCol == simWidth){ // row of quads done
			idxCol = 0;
			idxRow++;
		}
	}
	std::cout<<"happened5"<<std::endl;

	for (int i = 0; i < colorIndexCount; i++){
		colors[i].x = 0.8f;
		colors[i].y = 0.5f; 
		colors[i].z = 0.5f;
	}

		// Quad Position Buffer Array

	pointsVBO = 0;
	glGenBuffers(1, &pointsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glBufferData(GL_ARRAY_BUFFER, colorIndexCount*sizeof(float3), quadPoints, GL_STATIC_DRAW);

		// Color Buffer Array

	colorsVBO = 0;
	glGenBuffers(1, &colorsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glBufferData(GL_ARRAY_BUFFER, colorIndexCount*sizeof(float3), colors, GL_DYNAMIC_DRAW);

		// Cuda / openGL interop

	checkCuda(cudaGraphicsGLRegisterBuffer(&cudaColorResource, colorsVBO, cudaGraphicsMapFlagsNone));

		// vao binding

	GLuint vertexArray = 0;
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	glBindBuffer(GL_ARRAY_BUFFER, pointsVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	std::cout<<"happened7"<<std::endl;

			/////////*** Create shader program ***/////////

	GLuint shaderProgram = glCreateProgram();
	Link(shaderProgram);
	std::cout<<"happened8"<<std::endl;
	
}

void GLInstance::Link(GLuint& shaderProgram)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderText, NULL);
	glCompileShader(vertexShader);
	int params = -1;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &params);
	if(GL_TRUE != params){	
		int actual_length = 0;
		char log[2048];
		fprintf(stderr, "ERROR: GL shader idx %i did not compile\n", vertexShader);
		glGetShaderInfoLog(vertexShader, 500, &actual_length, log);
		std::cout << log;
		exit(1);
	}
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderText, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &params);
	if(GL_TRUE != params){	
		int actual_length = 0;
		char log[2048];
		fprintf(stderr, "ERROR: GL shader idx %i did not compile\n", vertexShader);
		glGetShaderInfoLog(vertexShader, 500, &actual_length, log);
		std::cout << log;
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
}
void GLInstance::Draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(shaderProgram);
	glBindVertexArray(vertexArray);
	glDrawArrays(GL_QUADS, 0, colorIndexCount);

	glfwSwapBuffers(window);
	glfwPollEvents();
}
GLInstance::~GLInstance()
{
	glDeleteBuffers(1, &pointsVBO);
	glDeleteBuffers(1, &colorsVBO);
	glDeleteVertexArrays(1, &vertexArray);
	glfwTerminate();
}
