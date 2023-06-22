# Pipeline

Cluster Optimisation by sampling of source

Optimal clustering by sampling the number of individuals 5 time adn lookign what is the more stable. By looking at the Adjuster Rand Index. The set of parameters the best best Adjusted Rand Index is then considered the best to use as the clustering is the most stable

```bash

# Preprocess the data
python src/run_clean_occupation.py
python src/run_data_to_sqlite.py

# Get the optimal Clustering
python src/run_optimal_clustering
python src/run_visu_optimization

# Run the graph
python src/run_graph.py
python src/run_graph_non_europe.py

# Analysis
python src/run_region_gini.py
python src/run_region_similarity.py

```
