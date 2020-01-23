default: pdf

pdf:
	cd paper/ ; latexmk -pdf -pdflatex=pdflatex main.tex
