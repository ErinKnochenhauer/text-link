# Functions for clustering
from functools import partial
import hdbscan
from hyperopt import fmin, tpe, STATUS_OK, space_eval, Trials
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer


def validate_unique_text(unique_text):
    if not isinstance(unique_text, list):
        raise ValueError("A list of unique text is required.")


def validate_hspace(hspace):
    if not isinstance(hspace, dict):
        raise ValueError("Search space should be a dictionary.")


def _init_sentence_transformers():
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Max Sequence Length:", st_model.max_seq_length)
    return st_model


def _encode_text(st_model, unique_text):
    return st_model.encode(unique_text)


def generate_clusters(embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      cluster_selection_epsilon,
                      random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state).fit_transform(embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               cluster_selection_epsilon=cluster_selection_epsilon,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)
    return clusters


def _score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)

    return label_count, cost


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify
    """
    clusters = generate_clusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 cluster_selection_epsilon= params['cluster_selection_epsilon'],
                                 random_state = params['random_state'])
    
    label_count, cost = _score_clusters(clusters, prob_threshold = 0.05)
    
    #15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0
    
    loss = cost + penalty
    
    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperopt hyperparameter space to minimize objective function
    """

    trials = Trials()
    fmin_objective = partial(objective,
                             embeddings=embeddings,
                             label_lower=label_lower,
                             label_upper=label_upper)

    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      cluster_selection_epsilon=best_params['cluster_selection_epsilon'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials


def save_cluster_results(unique_text, clusters, results_path=None):
    """Create a results df for auditing purposes and downstream processing"""

    docs_df = pd.DataFrame(unique_text, columns=["Doc"])
    docs_df['ClusterID'] = clusters.labels_
    docs_df['probability'] = clusters.probabilities_
    docs_df['Doc_ID'] = range(len(docs_df))

    # sort by cluster assignment
    docs_df.sort_values('ClusterID', ascending=True, inplace=True)

    if results_path:
        docs_df.to_csv(results_path)

    return docs_df


def _create_visualization_df(unique_text, embeddings, clusters):
    """
    Reduce dimensionality of embeddings to 3D and create a df for visualization input
    """

    # define umap projection configurations
    umap_3d_projection = umap.UMAP(n_neighbors=5,
                                   n_components=3,
                                   metric='cosine',
                                   random_state=42)

    # reduce embeddings to 3d for visualization
    projected_vectors = umap_3d_projection.fit_transform(embeddings)

    # create df for visualization formatting
    viz_df = pd.DataFrame(
        {
            'text': unique_text,
            'cluster': clusters.labels_,
            'd1': projected_vectors[:, 0],
            'd2': projected_vectors[:, 1],
            'd3': projected_vectors[:, 2]
        }
    )

    return viz_df


def visualize_clusters(unique_text, embeddings, clusters, remove_outliers=False):
    """
    Create a 3D visual to investigate cluster results.
    """
    viz_df = _create_visualization_df(unique_text, embeddings, clusters)

    if remove_outliers:
        viz_df = viz_df[viz_df.cluster != -1]

    fig = px.scatter_3d(viz_df,
                        x='d1', y='d2', z='d3',
                        color='cluster',
                        color_continuous_scale=px.colors.sequential.Rainbow,
                        width=1000, height=800,
                        hover_name='cluster',
                        hover_data={
                            'd1': False,  # remove d1 from hover data,
                            'd2': False,
                            'd3': False,
                            'text': True,
                            'cluster': False
                            }
                        )
    
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()


def cluster_text(unique_text,
                 hspace,
                 max_evals,
                 label_lower,
                 label_upper,
                 results_path=None):
    """
    Main function to embed text using pretrained Sentence Transformer model, reduce
    dimensionality with UMAP, perform density clustering with HDBSCAN, and finally 
    save the results and optionally visualize results.
    """
    
    validate_unique_text(unique_text)

    validate_hspace(hspace)

    # load pretrained sentence transformer model
    model = _init_sentence_transformers()

    # encode text using pretrained model
    embeddings = _encode_text(model, unique_text)

    # find the optimal cluster parameters using a Bayesian search
    best_params, best_clusters, trials_use = bayesian_search(embeddings,
                                                             space=hspace,
                                                             max_evals=max_evals,
                                                             label_lower=label_lower,
                                                             label_upper=label_upper)

    # cluster text using optimal parameters found in search
    clusters = generate_clusters(embeddings,
                                 n_neighbors=best_params['n_neighbors'],
                                 n_components=best_params['n_components'],
                                 min_cluster_size=best_params['min_cluster_size'],
                                 cluster_selection_epsilon=best_params['cluster_selection_epsilon'],
                                 random_state=best_params['random_state'])

    # visualize results
    visualize_clusters(unique_text, embeddings, clusters, remove_outliers=True)

    # get a results dataframe for auditing purposes
    results = save_cluster_results(unique_text, clusters, results_path=results_path)

    return results
