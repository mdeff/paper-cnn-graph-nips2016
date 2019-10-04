ODG = $(wildcard *.odg)
PDF = $(ODG:.odg=.pdf)

all: runtime.pdf $(PDF)

%.pdf: %.odg
	libreoffice --headless \
		"-env:UserInstallation=file:///tmp/libreoffice_conversion_${USER}" \
		--convert-to pdf $<
	pdfcrop $@ $@

clean:
	rm -f runtime.pdf
	rm -f $(PDF)

runtime.pdf: runtime.py
	./runtime.py
	pdfcrop runtime.pdf runtime.pdf

.PHONY: all clean
