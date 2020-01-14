default: pdf

pdf:
	cd paper/ ; latexmk -pdf -pdflatex=pdflatex plos-paper.tex
