PDF  = $(patsubst %.tex,%.pdf,$(wildcard *.tex))
BIB  = $(wildcard *.bib)

all: $(PDF)

images:
	$(MAKE) -C images

%.pdf: %.tex $(BIB) images
	latexmk -pdf $<

clean:
	$(MAKE) clean -C images
	rm -f *.{aux,bbl,blg,fdb_latexmk,fls,log,out,lof,dvi}
	rm -f *.{bcf,run.xml}
	rm -f *.{toc,snm,nav}
	rm -f *.gz*
	rm -f **/*eps-converted-to.pdf

cleanall: clean
	rm -f $(PDF)

.PHONY: all images clean cleanall
