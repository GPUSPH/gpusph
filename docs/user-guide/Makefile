SRCDIR1=gpusph-install
SRCDIR2=gpusph-setup
SRCDIR3=gpusph-theory
SRCDIR4=gpusph-valid

all: gpusph-install.pdf gpusph-setup.pdf gpusph-theory.pdf gpusph-valid.pdf

gpusph-install.pdf:
	cd $(SRCDIR1); make
	mv $(SRCDIR1)/$@ .

gpusph-setup.pdf:
	cd $(SRCDIR2); make
	mv $(SRCDIR2)/$@ .

gpusph-theory.pdf:
	cd $(SRCDIR3); make
	mv $(SRCDIR3)/$@ .

gpusph-valid.pdf:
	cd $(SRCDIR4); make
	mv $(SRCDIR4)/$@ .

clean:
	cd $(SRCDIR1); make clean
	cd $(SRCDIR2); make clean
	cd $(SRCDIR3); make clean
	cd $(SRCDIR4); make clean
	rm -rf gpusph-install.pdf
	rm -rf gpusph-setup.pdf
	rm -rf gpusph-theory.pdf
	rm -rf gpusph-valid.pdf

.PHONY: clean

.NOTPARALLEL:
