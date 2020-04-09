################################################
SRC = DauceAlbigesPerrinet2020
# SRC_rev = Pasturel_etal2019
DIR_rev = revision_0
SRC_rev = $(DIR_rev)/DauceAlbigesPerrinet2020
################################################
default: pdf diff
pdf: $(SRC).pdf
diff: $(SRC)_trackedchanges.pdf
################################################
LATEXMK = latexmk -bibtex -pdf
#  -pdflatex=pdflatex
################################################

$(SRC).pdf: $(SRC).tex Bibliography.bib
	$(LATEXMK) $(SRC).tex

# post-production
$(SRC)_trackedchanges.pdf: $(SRC).tex Bibliography.bib $(SRC_rev).tex
	latexdiff --flatten --graphics-markup=both $(SRC_rev).tex $(SRC).tex > $(SRC)_trackedchanges.tex
	$(LATEXMK) $(SRC)_trackedchanges.tex
	# open $(SRC)_trackedchanges.pdf

touch:
	touch *.tex
