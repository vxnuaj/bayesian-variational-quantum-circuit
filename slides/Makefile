all: main.pdf

main.pdf: main.tex ../figures/*.png
	tectonic main.tex

clean:
	rm -f main.pdf main.aux main.log main.nav main.out main.snm main.toc

.PHONY: all clean
