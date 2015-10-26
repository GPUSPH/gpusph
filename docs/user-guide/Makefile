SRCDIR=src

gpusph-manual.pdf:
	cd $(SRCDIR); make
	mv $(SRCDIR)/$@ .

clean:
	cd $(SRCDIR); make clean
	rm -rf gpusph-manual.pdf

.PHONY: clean gpusph-manual.pdf
