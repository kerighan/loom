# Minimal Sphinx makefile
SPHINXBUILD ?= python -m sphinx
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean

help:
	@echo "make html   - build the HTML docs into _build/html"
	@echo "make clean  - remove _build"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Open $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)"
