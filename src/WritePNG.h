#include <png.h>
#include <string.h>
#include <iostream>
#include <memory>

int WritePNG(unsigned char* data, std::string fileName, unsigned int width, unsigned int height);

std::string FrameNameGen(int frameIdx, int totalFrames);