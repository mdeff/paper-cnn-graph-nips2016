all: runtime.pdf illustrationCNNgraphs.pdf

runtime.pdf: runtime.py
	./runtime.py
	pdfcrop runtime.pdf runtime.pdf

illustrationCNNgraphs.pdf: illustrationCNNgraphs.eps
	repstopdf illustrationCNNgraphs.eps

clean:
	rm -f runtime.pdf
	rm -f illustrationCNNgraphs.pdf

.PHONY: all clean
