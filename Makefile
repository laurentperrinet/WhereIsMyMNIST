default: pdf

pdf:
	cd paper/ ; latexmk -pdf -pdflatex=pdflatex paper.tex
