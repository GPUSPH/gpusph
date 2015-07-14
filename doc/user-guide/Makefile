gpusph-manual.pdf:

%.pdf: %.tex
	pdflatex $<
	bibtex $*
	pdflatex $<
	pdflatex $<

gpusph-manual.pdf: gpusph-license.tex gpusph-reference.bib
