# Full Pipeline

jupyter:
	python -m jupyterlab

data_to_db:
	python run_data_to_sqlite.py

optimization:
	python run_optimal_clustering.py

visu_optimization:
	python run_visu_optimization.py

graph:
	python run_graph.py

graph_region:
	python run_graph_region.py

graph_region_non_europe:
	python run_graph_non_europe.py

region_similarity:
	python run_region_similarity.py

region_gini:
	python run_region_gini.py