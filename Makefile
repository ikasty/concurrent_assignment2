all:
	@echo "usage:"
	@echo "  make naive [options]      - use only 1 thread"
	@echo "  make cuda [options]    - use CUDA"
	@echo ""
	@echo "options:"
	@echo "  DEBUG=1        - set debug mode"
	@echo "  NORAND=1       - not using random value"

clean:
	rm -f assignment2

ifeq ($(DEBUG), 1)
OPTION += -DDEBUG_ENABLED
endif
ifeq ($(NORAND), 1)
OPTION += -DNORAND
endif

naive: assignment2.c
	@gcc assignment2.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DNAIVE -lm -g -O3 -o assignment2