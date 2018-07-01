CU_FILES := $(wildcard src/*.cu)
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)) $(notdir $(CPP_FILES:.cpp=.o)))
$(info OBJ_FILES is $(OBJ_FILES))
OBJDIR = obj

LIBS = -lglfw -lGL -lGLEW

CC = g++
CFLAGS = --std=c++11

NVCC = nvcc
NVFLAGS = -arch=sm_30 --std=c++11

#vmc: $(OBJ_FILES) 
#	$(NVCC) $(NVFLAGS) -o vmc $(OBJDIR)/main.o $(LIBS)

#obj/%.o: $(CU_FILES)
#	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

vmc: $(OBJ_FILES)
	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS) -lpng

obj/main.o: src/main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/VMC.o: src/VMC.cu
	$(NVCC) $(NVFLAGS) -c -dc $< -o $@  $(LIBS)

obj/GLInstance.o: src/GLInstance.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/WritePNG.o: src/WritePNG.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -lpng

obj/FormPNGData.o: src/FormPNGData.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/Zero_Histogram.o: src/Zero_Histogram.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/Color.o: src/Color.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/ExpectationEnergy.o: src/ExpectationEnergy.cu
	$(NVCC) $(NVFLAGS) -c -dc $< -o $@ $(LIBS)

obj/MCTest.o: src/MCTest.cu
	$(NVCC) $(NVFLAGS) -c -dc $< -o $@ $(LIBS)

obj/Step.o: src/Step.cu
	$(NVCC) $(NVFLAGS) -c -dc $< -o $@ $(LIBS)

obj/PopulateBins.o: src/PopulateBins.cu
	$(NVCC) $(NVFLAGS) -c -dc $< -o $@ $(LIBS)

clean: 
	rm -f obj/*.o




