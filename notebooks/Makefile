# default: 0_parameters.ipynb 1_problem_statement.ipynb 2_methods.ipynb
default: 3_robust-What-training.ipynb 3_results_figure.ipynb 3_results_figure_robust.ipynb

IPYNB_CMD=jupyter nbconvert --ExecutePreprocessor.timeout=0 --allow-errors --execute --to notebook --inplace
%.ipynb: ../.git
	$(IPYNB_CMD)  $@

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

weigth:
	du -h . --max-depth=1

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402
