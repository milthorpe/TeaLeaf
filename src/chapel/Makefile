SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules

# -------

CHPL_FLAGS+=-senableProfiling=true -sprintLocaleInfo=true --detailed-errors --savec tmp --print-commands
CHPL_SOURCE=$(wildcard *.chpl)
ifeq ($(CHPL_LOCALE_MODEL),gpu)
# GPU version
ifndef CHPL_GPU
$(error CHPL_GPU not defined)
else
BLOCK_SIZE=512
CHPL_FLAGS+=--report-gpu -suseGPU=true --ldflags -no-pie --gpu-block-size=$(BLOCK_SIZE)
ifeq ($(CHPL_GPU),nvidia)
LDFLAGS+=-L$(CUDA_PATH)/lib64 -lcudart
else
ifeq ($(CHPL_GPU),amd)
LDFLAGS+=-lamdhip64
else
$(error Unknown value for CHPL_GPU: $(CHPL_GPU))
endif
endif
ifndef CHPL_GPU_ARCH
$(error CHPL_GPU_ARCH not defined)
endif
endif
else
# CPU version
ifdef CHPL_GPU
$(error CHPL_GPU is set, but CHPL_LOCALE_MODEL!= gpu! CHPL_LOCALE_MODEL==$(CHPL_LOCALE_MODEL))
endif
endif
CHPL_FLAGS+=--fast --no-ieee-float# --llvm-remarks=vector
CUDACXX_FLAGS+=-O3
ifdef DEBUG
  CHPL_FLAGS+=-g
  CUDACXX_FLAGS=-g
endif
CHPL = chpl

# -------
EXE = chapel-tealeaf
.PHONY: all $(EXE) clean

all: $(EXE)

$(EXE): $(CHPL_SOURCE)
	$(CHPL) $(CHPL_FLAGS) $(CHPL_SOURCE) -o $@

$(EXE)-block: $(CHPL_SOURCE)
	$(CHPL) $(CHPL_FLAGS) -suseBlockDist=true $(CHPL_SOURCE) -o $@

$(EXE)-stencil: $(CHPL_SOURCE)
	$(CHPL) $(CHPL_FLAGS) -suseStencilDist=true $(CHPL_SOURCE) -o $@

clean:
	rm -f $(EXE) $(EXE)-block $(EXE)-stencil

