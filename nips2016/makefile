PDF := $(patsubst %.tex, %.pdf, $(wildcard *.tex))
SRC := $(filter-out %_crop.pdf, $(wildcard images/*.pdf))
FIG := $(patsubst %.pdf, %_crop.pdf, $(SRC))

.SECONDARY: $(FIG)

all: $(PDF)

%.pdf: %.tex $(FIG)
	latexmk -pdf $<

%_crop.pdf: %.pdf
	pdfcrop $< $@

clean:
	rm -f *.{aux,bbl,blg,fdb_latexmk,fls,log,out}
	rm -f *.{bcf,run.xml}
	rm -f $(FIG)
