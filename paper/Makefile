all: main.pdf

main.pdf: main.tex refs.bib ../figures/*.png
	tectonic main.tex

clean:
	rm -f main.pdf main.bbl main.aux main.log main.blg main.out

.PHONY: all clean
