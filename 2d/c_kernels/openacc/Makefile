vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built OpenACC kernels"
	-mv *.lst *.ptx *.cub obj/$(KERNELS)/

ifeq ($(OPENACC_TARGET), HSW)
    CFLAGS += -ta=multicore -tp=haswell-64 
endif
ifeq ($(OPENACC_TARGET), IVB)
    CFLAGS += -ta=multicore -tp=ivybridge-64 
endif
ifeq ($(OPENACC_TARGET), GPU)
    CFLAGS += -ta=tesla,cc35,kepler -tp=ivybridge-64
endif
ifndef OPENACC_TARGET
    CFLAGS += -ta=tesla,cc35,kepler -tp=ivybridge-64
endif

CFLAGS += -acc -Minfo=accel -DDIFFUSE_OVERLOAD

