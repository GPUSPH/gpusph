DIRS=gpusph-install \
     gpusph-setup \
     gpusph-theory \
     gpusph-valid

TARGETS=$(patsubst %,%.pdf,$(DIRS))

all: $(TARGETS)

%.pdf:
	make -C $(basename $@)
	mv $(basename $@)/$@ $@

clean:
	for dir in $(DIRS) ; do make -C $$dir clean ; done
	rm -rf $(TARGETS)

.PHONY: clean

.NOTPARALLEL:
