TEX = $(wildcard *.tex)
PDF = $(TEX:.tex=.pdf)
BIB = $(wildcard *.bib)

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
	rm -f arxiv.zip

cleanall: clean
	rm -f $(PDF)

arxiv: all
	apack arxiv.zip images/*.pdf $(TEX:.tex=).{tex,sty,bbl}

.PHONY: all images clean cleanall arxiv
