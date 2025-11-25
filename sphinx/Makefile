SPHINXBUILD ?= sphinx-build
SOURCEDIR = source

copy_notebooks:
	cp -u ../notebooks/*.ipynb $(SOURCEDIR)/notebooks

html: copy_notebooks
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" ../docs