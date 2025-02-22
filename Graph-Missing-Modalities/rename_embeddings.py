import numpy as np
import pandas as pd
import os


textual_embeddings_folder = './data/datasets/textual_embeddings_05/sentence_transformers/sentence-transformers/all-mpnet-base-v2/2'

textual_embeddings_folder_indexed = './data/datasets/textual_embeddings_05/sentence_transformers/sentence-transformers/all-mpnet-base-v2/3'

interactions = pd.read_csv('interactions.tsv', sep='\t', header=None)

if not os.path.exists(textual_embeddings_folder_indexed):
    os.makedirs(textual_embeddings_folder_indexed)

for couple in interactions.values:
    try:
        np.save(f'{textual_embeddings_folder_indexed}/{couple[2]}.npy', np.load(f'{textual_embeddings_folder}/{couple[0]},{couple[1]}.npy'))
    except FileNotFoundError:
        continue