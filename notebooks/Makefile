default: METHODS # 3_robust-What-training.ipynb 3_results_figure.ipynb 3_results_figure_robust.ipynb 4_results_figure_contrast.ipynb 5_results_figure_closed_loop.ipynb 6_results_parameters.ipynb

METHODS: 0_parameters.ipynb 1_problem_statement.ipynb 2_methods.ipynb
# default: 3_robust-What-training.ipynb 3_results_figure.ipynb 3_results_figure_robust.ipynb
# default: 4_results_figure_contrast.ipynb 5_results_figure_closed_loop.ipynb 6_results_parameters.ipynb

IPYNB_CMD=jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace
%.ipynb: ../.git
	$(IPYNB_CMD)  $@


INK = inkscape  --without-gui --export-type=eps  --export-ignore-filters
convert_eps:
	mkdir -p figures_PLoS-CB/
	$(INK) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure1.pdf -o figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure1.eps
	$(INK) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure2.pdf -o figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure2.eps
	$(INK) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure3.pdf -o figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure3.eps
	$(INK) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure4.pdf -o figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure4.eps
	$(INK) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure5.pdf -o figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure5.eps

CONV = convert -density 600  -flatten  -compress lzw -units pixelsperinch  -depth 8
convert_tiff:
	mkdir -p figures_PLoS-CB/
	$(CONV) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure1.pdf figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure1.tiff
	$(CONV) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure2.pdf figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure2.tiff
	$(CONV) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure3.pdf figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure3.tiff
	$(CONV) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure4.pdf figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure4.tiff
	$(CONV) figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure5.pdf figures_PLoS-CB/Pasturel_etal2020_PLoS-CB-figure5.tiff



clean_models:
	rm */models/*.pt; rm */*/models/*.pt

weigth:
	du -h . --max-depth=1

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402
