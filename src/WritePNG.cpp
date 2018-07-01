#include "WritePNG.h"
int WritePNG(unsigned char* data, std::string fileName, unsigned int width, unsigned int height)
{

	FILE *pngFile = fopen(fileName.c_str(), "wb");
  std::cout << "creating png image: " << fileName << std::endl;
  png_structp png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, 
  											  nullptr,
  											  nullptr,
  											  nullptr);
  if (!png_ptr)
	fprintf(stderr, "ERROR: png_ptr not created\n");

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
     png_destroy_write_struct(&png_ptr,
       (png_infopp)NULL);
     fprintf(stderr, "ERROR: into_ptr not created\n");
  }

  png_init_io(png_ptr, pngFile);
 	png_set_IHDR(png_ptr, info_ptr, width, height,
  8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
  PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep)*height);
  for (int i = 0; i <  height; i++)
  {
    row_pointers[i] = data + i * 3 * width;
    //std::cout << "Row Pointer starts at val: " << i*width << " "<< (int)data[i*width] << std::endl;
  }

  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, row_pointers);

	if (pngFile != NULL) fclose(pngFile);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  // WARNING Calling code should free its own data pointer when ready to do so. 
  //if (data != NULL) free(data);		
}


std::string FrameNameGen(int frameIdx, int totalFrames)
{
	int numZeros = std::to_string(totalFrames).length() + 1 - std::to_string(frameIdx).length();
	std::string stepsString = std::string(numZeros, '0') + std::to_string(frameIdx);
	std::string fileName = "video/frame_" + stepsString + ".png";
	return fileName;
}
// int main()
// {
// 	int width = 500;
// 	int height = 500;
// 	std::cout << sizeof(png_byte) << std::endl;
// 	png_bytep data = (png_bytep) calloc(3 * (width) * (height), sizeof(png_bytep));
// 	int u = 0;
// 	int v = 0;
// 	int row, col;
// 	int magic = 0;
// 	int row_magic = 3;
// 	int width_magic = 0;
// 	int const_magic = 0;

// 	data[0] = 255;
// 	data[1] = 255;
// 	data[2] = 255;
// 	data[3] = 255;
// 	data[4] = 255;
// 	data[5] = 255;
// 	data[199] = 255;
// 	data[198] = 255;

// 	data[200] = 255;

// 	data[11] = 0;

// 	for (row = 0; row <= 3*height; row = row + 1)
// 	{
// 		for (col = 0; col <= width; col = col + 1)
// 		{

// 			int idx = row*width + col;
// 			if (idx % 3 == 0)
// 			{
// 				data[idx] = 255;
// 				data[idx+1] = 0;
// 				data[idx+2] = 255;
// 			}

// 			if (idx/3 > 200 && idx/3 < 5000)
// 			{
// 				data[idx] = 0;
// 				data[idx+1] = 0;
// 				data[idx+2] = 0;
// 			}
// 			if (col == 200)
// 			{
// 				data[idx] = 0;
// 				data[idx+1] = 0;
// 				data[idx+2] = 0;	
// 			}
// 			// if ((row*width + 3*col +2) > (3 * width * height))
// 			// {
// 			// 	std::cout << "HAPPENED " << std::endl;
// 			// }

// 			// std::cout << row << " " << col << std::endl;
// 			// std::cout << 3*row*width + 2 * width_magic + magic*col + 2 * const_magic << " should not be greater than " << 3 * width * height << std::endl;

// 			// data[1*row*width + magic*col] = 255;
// 			// data[2*row*width + width_magic + magic*col + 1 * const_magic] = 255;
// 			// data[3*row*width + 2 * width_magic + magic*col + 2 * const_magic] = 255;
// 			// if (col == 100)
// 			// {
// 			// 	data[1*row*(width) + magic*col] = 0;
// 			// 	data[2*row*(width) + width_magic + magic*col + 1 * const_magic] = 0;
// 			// 	data[3*row*(width) + 2 * width_magic + magic*col + 2 * const_magic] = 0;
// 			// 	v++;
// 			// }
// 			// if (row < (2*height)/4 && row > height/4 && col < (2*width)/4 && col > width/4)
// 			// {
// 			// 	data[row_magic*row*(width) + magic*col] = 0;
// 			// 	data[row_magic*row*(width) + width_magic + magic*col + 1 * const_magic] = 255;
// 			// 	data[row_magic*row*(width) + 2 * width_magic + magic*col + 2 * const_magic] = 0;
// 			// 	u++;
// 			// }
// 		}
// 	}	
// 	writeImage(data, height, width);
// 	return 0;
// }