CU_FILES := $(wildcard src/*.cu)
OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))

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
	$(NVCC) $(NVFLAGS) -o $@ $(OBJ_FILES) $(LIBS)

obj/main.o: src/main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

obj/VMC.o: src/VMC.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@  $(LIBS)

obj/GLInstance.o: src/GLInstance.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@ $(LIBS)

clean: 
	rm -f obj/*.o




