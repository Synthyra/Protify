import os
import torch
import esm
import re
from Bio import SeqIO
import networkx as nx
import numpy as np
import gc


class TokenToSequencePooler:
    def __init__(self, token_emb, attention_layers):
        # Initialize the pooler by loading token embeddings and attention layers.
        # Handles the removal of CLS and END token embeddings from representations.

        self.token_emb = token_emb
        self.attention_layers = attention_layers
        if self.token_emb is not None:
          if len(self.token_emb.shape) == 2:
            self.representations =  self.token_emb[1:-1]
          elif len(self.token_emb.shape) == 3:
            self.representations = self.token_emb[:,1:-1, :]
        else:
          self.representations = None

        self.attn_all_layers = self.attention_layers

    def create_pooled_matrices_across_layers(self, mtx_all_layers):
        # Perform max pooling across layers by selecting the maximum values across attention layers.
        # Returns the matrix after pooling the attention layers.

        mtx_max_of_max = torch.max(mtx_all_layers[1], dim=1)[0]
        return mtx_max_of_max

    def pool_parti(self,
                  verbose=False,
                  return_importance=False):
        # Perform pooling based on PageRank algorithm applied to attention matrices.
        # Optionally return importance weights or print details about the importance calculation.
        # Handles errors during the pooling process by printing detailed information.

        matrix_to_pool = self.create_pooled_matrices_across_layers(mtx_all_layers=self.attn_all_layers).squeeze().numpy()

        dict_importance = self._page_rank(matrix_to_pool)
        importance_weights = np.array(list(self._calculate_importance_weights(dict_importance).values()))

        if return_importance:
            return importance_weights
        if verbose:
            print(f'pagerank direct outcome is {dict_importance}\n')
            print(f'importance_weights dict of length {len(importance_weights)} looks like\n {importance_weights}')
            print(f'shape of the importance matrix is {len(importance_weights)} and for repr, its {self.representations.shape}')
            print(f"importance weights look like {sorted(importance_weights, reverse=True)[0:5]}")

        try:
            return torch.tensor(np.average(self.representations, weights=importance_weights, axis=0))
        except Exception as e:
            print(f"{e} in PageRank without cls", flush=True)
            print(f"self.representations shape {self.representations.shape}", flush=True)
            print(f"importance_weights {len(importance_weights)}", flush=True)
            return None

    def _page_rank(self, attention_matrix, personalization=None, nstart=None, prune_type="top_k_outdegree"):
        # Run PageRank on the attention matrix converted to a graph.
        # Raises exceptions if the graph doesn't match the token sequence or has no edges.
        # Returns the PageRank scores for each token node.

        G = self._convert_to_graph(attention_matrix)

        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.")
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(G, alpha=0.85, tol=1e-06, weight='weight', personalization=personalization,
                           nstart=nstart, max_iter=100)

    def _convert_to_graph(self, matrix):
        # Convert a matrix (e.g., attention scores) to a directed graph using networkx.
        # Each element in the matrix represents a directed edge with a weight.

        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

        return G

    def _calculate_importance_weights(self, dict_importance):
        # Normalize the PageRank scores (importance values) so they sum to 1.
        # Exclude CLS and END token from the importance calculation.

        # Get the highest integer key
        highest_key = max(dict_importance.keys())

        # Remove the entry with the highest key (END) and the entry with key 0 (CLS token)
        del dict_importance[highest_key]
        del dict_importance[0]

        total = sum(dict_importance.values())
        return {k: v / total for k, v in dict_importance.items()}

def main_pooling(token_emb, 
                 attention_layers, 
                 ):
    """
    Main function to perform pooling operations on protein sequence data.

    Args:
        token_emb (str): Token embeddings
        attention_layers (str): Attention matrices
    """
    # # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    # file_name = os.path.basename(token_emb)

    # Instantiate the TokenToSequencePooler
    pooler = TokenToSequencePooler(token_emb=token_emb,
                                    attention_layers=attention_layers)
    if pooler.token_emb is None or pooler.attn_all_layers is None:
        print(f"Skipping pooling due to missing data.")
        return

    rep_w_cls = pooler.token_emb
    attn = pooler.attn_all_layers

    # Check if the shapes of representations and attentions match
    if not rep_w_cls.shape[0] == attn.shape[-1]:
        if len(rep_w_cls.shape) == 3 and not rep_w_cls.shape[1] == attn.shape[-1]:
            print(f"The attention and representation shapes don't match", flush=True)
            return

    # Perform Pool PaRTI pooling

    pooled = pooler.pool_parti(verbose=False, return_importance=False)
    return pooled