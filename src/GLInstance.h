#pragma once

// openGL includes
#include <GL/glew.h>
#define GLFW_DLL
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>

// CUDA includes
#include <vector_types.h>
#include <cuda_gl_interop.h>

// std includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <sstream>

inline 
cudaError_t checkCuda(cudaError_t result){
	if (result != cudaSuccess){
		fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
		assert(result==cudaSuccess);
	}
	return result;
}


class GLInstance
{
	public: 
		GLInstance(int width, int height);

		void InitWindow(int width, int height);

		void InitShaders(const char *vertexShaderText, const char *fragmentShaderText, GLuint shaderProgram);
				    	 
		void Link(GLuint& shaderProgram);

		void Draw();

		~GLInstance();

		GLFWwindow* window;
		GLuint shaderProgram;
		GLuint vertexArray;
		GLuint colorsVBO;
		GLuint pointsVBO;
		float3* quadPoints;
		float3* colors;
		GLuint vertexShader;
		GLuint fragmentShader;
		int colorIndexCount;
		struct cudaGraphicsResource *cudaColorResource;

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
