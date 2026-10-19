# Paper figures: one standalone .tex per figure, every number read from data/*.csv.
#
#   make            build every figure (build/*.pdf) from the committed data/
#   make data       re-export data/ from ../../results (needs the pipeline outputs; see README.md here)
#   make check      exit 1 if any figure's check failed (data/checks.csv)
#   make gray       greyscale renders in build/gray/ (every figure must still read without colour)
#   make png        300 dpi colour previews in build/png/
#   make preview    build/preview.pdf: every figure unscaled on the paper's page, with draft captions
#   make clean
#
# A figure whose check failed still builds, with a red stamp naming the check: it is never silently polished.
SHELL   := /bin/bash
FIGS    := $(sort $(basename $(wildcard fig*.tex)))
PDFS    := $(addprefix build/,$(addsuffix .pdf,$(FIGS)))
DATA    := $(wildcard data/*)
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build
RESULTS ?= ../../results
PYTHON  ?= python

.PHONY: all data check gray png preview clean
all: $(PDFS)

build/%.pdf: %.tex figstyle.tex notation.tex $(DATA)
	@mkdir -p build
	@$(LATEXMK) $< > build/$*.out 2>&1 || { tail -40 build/$*.out; echo "*** $< failed: build/$*.log"; exit 1; }
	@echo "built $@  ($$(pdfinfo $@ | awk '/Page size/{printf "%.2f x %.2f in", $$3/72, $$5/72}'))"

data:
	cd ../../code && $(PYTHON) export_figdata.py --results $(abspath $(RESULTS)) --out $(abspath data)

check: $(PDFS)
	@test -f data/checks.csv || { echo "no data/checks.csv: run make data"; exit 1; }
	@bad=0; for f in $(PDFS); do w=$$(pdfinfo $$f | awk '/Page size/{print $$3/72}'); \
	  if awk "BEGIN{exit !($$w > 6.5 + 0.005)}"; then echo "TOO WIDE: $$f is $$w in (text width 6.5 in)"; bad=1; fi; done; \
	 if grep -q ',FAIL$$' data/checks.csv; then echo "FAILED checks:"; grep ',FAIL$$' data/checks.csv; bad=1; fi; \
	 if [ $$bad = 0 ]; then echo "all figure checks passed; every figure fits the 6.5 in text width"; fi; exit $$bad

gray: $(PDFS)
	@mkdir -p build/gray
	@for f in $(FIGS); do pdftoppm -r 200 -gray -png -singlefile build/$$f.pdf build/gray/$$f; done
	@echo "greyscale renders in build/gray/"

png: $(PDFS)
	@mkdir -p build/png
	@for f in $(FIGS); do pdftoppm -r 300 -png -singlefile build/$$f.pdf build/png/$$f; done

preview: build/preview.pdf
build/preview.pdf: preview.tex notation.tex $(wildcard build/fig*.pdf)
	@$(LATEXMK) preview.tex > build/preview.out 2>&1 || { tail -40 build/preview.out; exit 1; }
	@echo "built $@ ($$(pdfinfo $@ | awk '/Pages/{print $$2}') pages)"

clean:
	rm -rf build
