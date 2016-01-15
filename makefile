SRC = $(wildcard *.md)
PDF = $(SRC:.md=.pdf)

all: $(PDF)

%.tex: %.md
	pandoc -o $@ $< \
		-s \
		-N \
		--natbib \
		--biblio /data/research/bibliography/refs.bib \
		-V biblio-style=plain \
		-V lang=en \
		-V documentclass=article \
		-V papersize=a4 \
		-V geometry=vmargin=3cm,hmargin=3cm \
		-V fontsize=11pt \
		--filter pandoc-crossref

%.pdf: %.tex
	pdflatex -interaction errorstopmode $<
	bibtex $(*F)
	pdflatex -interaction errorstopmode $<
	pdflatex -interaction errorstopmode $<
	rm -f *.aux *.log *.toc *.bbl *.blg *.out

clean:
	rm -f *.tex *.pdf
