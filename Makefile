all:
	@echo "usage:"
	@echo "  make naive [options]      - use only 1 thread"
	@echo "  make cuda [options]       - use CUDA"
	@echo "  make info                 - make execution file only for get information"
	@echo ""
	@echo "options:"
	@echo "  DEBUG=1        - set debug mode"
	@echo "  NORAND=1       - not using random value"

clean:
	rm -f assignment2 assignment2.c

ifeq ($(DEBUG), 1)
OPTION += -DDEBUG_ENABLED
endif
ifeq ($(NORAND), 1)
OPTION += -DNORAND
endif

naive: assignment2.cu
	@cp assignment2.cu assignment2.c
	@gcc assignment2.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DNAIVE -lm -g -O3 -o assignment2

cuda: assignment2.cu
	nvcc assignment2.cu -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DCUDA -L/usr/local/cuda/lib -lcudart -lm -g -O3 -o assignment2

info: assignment2.cu
	nvcc assignment2.cu -D_SVID_SOURCE -D_XOPEN_SOURCE=600 -DINFO -L/usr/local/cuda/lib -lcudart -lm -g -O3 -o assignment2
