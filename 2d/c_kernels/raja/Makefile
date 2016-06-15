RAJA_PATH 	 		 = $(KERNELS_PATH)/raja_src/
RAJA_SRC 	 		 = $(RAJA_PATH)/src/
RAJA_INC 	 		 = $(RAJA_PATH)/include/
RAJA_PLATFORM        = -DRAJA_PLATFORM_X86_AVX
RAJA_COMPILER_INTEL  = -DRAJA_COMPILER_ICC
RAJA_COMPILER_GNU    = -DRAJA_COMPILER_GNU
RAJA_COMPILER		:= $(RAJA_COMPILER_$(COMPILER))

OPTIONS += -DRAJA_BOX_SEGMENT

vpath %.cxx $(RAJA_SRC)
vpath %.cpp $(KERNELS_PATH)
vpath %.hpp $(KERNELS_PATH)

TL_LINK		 = $(CPP)
TL_COMPILER  = $(CPP)
TL_FLAGS 	 = $(CPPFLAGS) $(RAJA_COMPILER) $(RAJA_PLATFORM) -I$(RAJA_INC) \
			   $(RAJA_RULES)
TL_LDFLAGS	+= $(OMP)

# Currently a very constrained compilation
ifeq ($(CUDA), yes)
  OPTIONS      += -DNO_MPI
  KERNEL_CPP	= nvcc
  KERNEL_FLAGS  = -ccbin g++ -O2 --expt-extended-lambda -restrict \
  				  -arch compute_35 -std=c++11 -Xcompiler -fopenmp --x cu \
  				  $(RAJA_COMPILER) -DRAJA_USE_CUDA $(RAJA_PLATFORM) \
  				  $(RAJA_RULES) -I$(RAJA_INC) -I$(KERNELS_PATH)
  TL_LDFLAGS   += -lcudart -fopenmp -L/$(CUDA_PATH)/lib64
else
  KERNEL_CPP	= $(CPP)
  KERNEL_FLAGS  = $(TL_FLAGS) -I$(KERNELS_PATH)
endif

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.cpp)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.cpp, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))
RAJA_SRCS   := $(wildcard $(RAJA_SRC)/*.cxx)
KERNEL_OBJS += $(patsubst $(RAJA_SRC)/%.cxx, obj/$(KERNELS)/%.o, $(RAJA_SRCS))

include $(KERNELS_PATH)/make.deps
include $(RAJA_PATH)/build/RAJA_rules.mk

build_kernels: $(KERNEL_OBJS) Makefile
	@echo "Built RAJA kernels"

obj/$(KERNELS)/%.o: %.cpp Makefile make.deps $(KERNELS_PATH)/make.deps
	$(KERNEL_CPP) $(KERNEL_FLAGS) $(OPTIONS) -c $< -o $@

obj/$(KERNELS)/%.o: %.cxx Makefile make.deps $(KERNELS_PATH)/make.deps
	$(KERNEL_CPP) $(KERNEL_FLAGS) $(OPTIONS) -c $< -o $@

# Treat the chunk.c as an independent C++ source file for interfacing
obj/$(KERNELS)/chunk.o: chunk.c Makefile make.deps
	$(KERNEL_CPP) $(KERNEL_FLAGS) $(OPTIONS) -c $< -o $@

