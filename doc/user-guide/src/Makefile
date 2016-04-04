LTX ?= pdflatex
LTXFLAGS ?=

# Command to convert an SVG to PDF
SVG2PDF ?= inkscape -A $@ $<

EXTRA_DEPS = $(patsubst %.svg, %.pdf, $(wildcard fig/*.svg))

gpusph-manual.pdf:

# Set of general pattern rules to produce final and intermediate results
# during a compilation.
#
# The chain is the following:
# * the final PDF is generated from a .tex file, but it also depends on the
#   .bbl file produced by bibtex;
# * the .bbl is produced by a .bib file reading from an .aux file
# * the .aux file is produced by processing the .tex file once, and there
#   should also be a .bib file;
# * in the end, this creates the correct chain of dependencies to regenerate
#   the whole document whenever the source .tex or .bib are changed;

# Notes and TODOs: currently this (1) requires a .bib file and (2) requires
# the .bib file to be called the same as the .tex source. We would like to
# overcome these restrictions somehow (e.g. parse the .tex files to get the
# nme of the bib file could be an idea)

%.pdf: %.tex %.aux %.bbl
	$(LTX) $(LTXFLAGS) $< && \
	$(LTX) $(LTXFLAGS) $< && \
	while grep -Ei 'Warning.*Rerun' $(subst .tex,.log,$<) ; do $(LTX) $(LTXFLAGS) $< ; done

%.aux: %.tex %.bib $(EXTRA_DEPS)
	$(LTX) $(LTXFLAGS) $<

%.bbl: %.aux %.bib
	bibtex $<

fig/%.pdf: fig/%.svg
	$(SVG2PDF)

#gpusph-manual.pdf: gpusph-license.tex gpusph-reference.bib

clean:
	rm -rf *.aux *.bbl *.log *.out *.blg *.toc *.pdf fig/*pdf

.PHONY: clean
