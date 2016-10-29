PDF  = $(patsubst %.tex,%.pdf,$(wildcard *.tex))
BIB  = $(wildcard *.bib)

all: images $(PDF)

images:
	cd images && \
	./runtime.py && \
	pdfcrop runtime.pdf runtime_crop.pdf && \
	rm runtime.pdf

%.pdf: %.tex $(BIB)
	latexmk -pdf $<

clean:
	rm -f *.{aux,bbl,blg,fdb_latexmk,fls,log,out,lof,dvi}
	rm -f *.{bcf,run.xml}
	rm -f *.{toc,snm,nav}
	rm -f *.gz*
	rm -f **/*eps-converted-to.pdf
	rm -f images/runtime_crop.pdf

cleanall: clean
	rm -f $(PDF)

.PHONY: all images clean cleanall
