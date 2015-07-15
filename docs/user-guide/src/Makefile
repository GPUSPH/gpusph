gpusph-manual.pdf:

%.pdf: %.tex
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

gpusph-manual.pdf: gpusph-license.tex gpusph-reference.bib

clean:
	rm -rf *.aux *.bbl *.log *.out *.blg *.toc

.Phony: clean
