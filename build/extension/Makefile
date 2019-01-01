SUBDIRS := $(wildcard */.)

.DEFAULT_GOAL=build

.DEFAULT:
	for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir $@; \
		fi \
	done
