SUBDIRS := $(wildcard */.)

build:
	for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir $@; \
		fi \
	done

install:
	for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir $@; \
		fi \
	done

clean:
	for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ]; then \
			$(MAKE) -C $$dir $@; \
		fi \
	done
