# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:52:08 2023

@author: utkid
"""

from __future__ import absolute_import, division, print_function
import data_preprocess
import numpy as np
from tqdm import tqdm
import pandas as pd
from my_knowledge_graph import *
from cafe_utils import *
from data_preprocess import *
import warnings
warnings.filterwarnings("ignore")
import osma
import csv
def feature_engineering(movie_df):
    def extract_features(movie_df):
        # Extracting genre features and creating a new dataframe with one-hot encoded genre columns
        genre_features = movie_df['genre'].str.get_dummies(sep='|')
        return genre_features
    
    # Applying the function
    genre_features = extract_features(movie_df)
    
    # Merging the genre_features dataframe with the original movie dataframe
    movie_df_extended = movie_df.merge(genre_features, left_index=True, right_index=True)
    
    return movie_df_extended


# Assuming ml1m_movies_df is defined in data_preprocess module
ml1m_movies_df = data_preprocess.ml1m_movies_df

# Applying the feature_engineering function and getting the extended movie dataframe
extended_movie_df = feature_engineering(ml1m_movies_df)

def calculate_diversity_score(interaction_matrix):
    # Calculate diversity score for each user
    diversity_scores = []
    for index, row in interaction_matrix.iterrows():
        pi = row / row.sum()
        #print("LOOOOOOOKKKKK!!!!")
        #print(f"Proportions for user {index}: {pi.values}")  # Debugging line
        diversity_score = -np.sum(pi * np.log(pi))
        diversity_scores.append(diversity_score)
    
    return diversity_scores



            
def load_kg_embedding(dataset: str):
    """Note that entity embedding is of size [vocab_size+1, d]."""
    print('>>> Load KG embeddings ...')
    state_dict = load_embed_sd(dataset)
    # print(state_dict.keys())
    embeds = dict()
    # Load entity embeddings
    for entity in ENTITY_LIST[dataset]:
        embeds[entity] = state_dict[entity + '.weight'].cpu().data.numpy()[:-1]   # remove last dummy embed with 0 values.
        print(f'>>> {entity}: {embeds[entity].shape}')
    for rel in RELATION_LIST[dataset]:
        embeds[rel] = (
            state_dict[rel].cpu().data.numpy()[0],
            state_dict[rel + '_bias.weight'].cpu().data.numpy()
        )
    return embeds


def compute_top100_items(dataset):
    embeds = load_embed(dataset)
    user_embed = embeds[USER]
    product_embed = embeds[PRODUCT]
    if dataset == ML1M:
        purchase_embed, purchase_bias = embeds[WATCHED]
    elif dataset == LFM1M:
        purchase_embed, purchase_bias = embeds[LISTENED]
    else:
        purchase_embed, purchase_bias = embeds[PURCHASE]
    scores = np.dot(user_embed + purchase_embed, product_embed.T)
    user_products = np.argsort(scores, axis=1)  # From worst to best
    best100 = user_products[:, -100:][:, ::-1]
    print(best100.shape)
    return best100


def estimate_path_count(args):
    kg = load_kg(args.dataset)
    num_mp = len(kg.metapaths)
    train_labels = load_labels(args.dataset, 'train')
    counts = {}
    pbar = tqdm(total=len(train_labels))
    for uid in train_labels:
        counts[uid] = np.zeros(num_mp)
        for pid in train_labels[uid]:
            for mpid in range(num_mp):
                cnt = kg.count_paths_with_target(mpid, uid, pid, 50)
                counts[uid][mpid] += cnt
        counts[uid] = counts[uid] / len(train_labels[uid])
        pbar.update(1)
    save_path_count(args.dataset, counts)



def main(args):
    # Create user-item interaction matrix
    # Assuming user_item_df is a DataFrame with columns 'user_id', 'genre' representing user-item interactions
    user_item_df = extended_movie_df.assign(genre=extended_movie_df['genre'].str.split('|')).explode('genre') #Replace with your actual column names
    interaction_matrix = pd.pivot_table(user_item_df, index='new_id', columns='genre', aggfunc=len, fill_value=0)

    
    # Calculate diversity scores
    diversity_scores = calculate_diversity_score(interaction_matrix)
 
    # Define file paths 
    embed_file_path = "../../data/ml1m/preprocessed/cafe/tmp/embed.pkl"
    kg_file_path = "../../data/ml1m/preprocessed/cafe/tmp/kg.pkl"
    user_products_file_path = "../../data/ml1m/preprocessed/cafe/tmp/user_products_pos.npy"
    path_count_file_path = "../../data/ml1m/preprocessed/cafe/tmp/path_count.pkl"

    # Run following code to extract embeddings from state dict.
    if not os.path.exists(embed_file_path):
        embeds = load_kg_embedding(args.dataset)
        save_embed(args.dataset, embeds)
    else:
        print(f"The file {embed_file_path} already exists, skipping generation.")

    # Run following codes to generate MyKnowledgeGraph object.
    if not os.path.exists(kg_file_path):
        kg = MyKnowledgeGraph(args.dataset)
        save_kg(args.dataset, kg)
    else:
        print(f"The file {kg_file_path} already exists, skipping generation.")

    # Run following codes to generate top100 items for each user.
    if not os.path.exists(user_products_file_path):
        best100 = compute_top100_items(args.dataset)
        save_user_products(args.dataset, best100, 'pos')
    else:
        print(f"The file {user_products_file_path} already exists, skipping generation.")

    # Run following codes to estimate paths count.
    if not os.path.exists(path_count_file_path):
        estimate_path_count(args)
    else:
        print(f"The file {path_count_file_path} already exists, skipping generation.")


if __name__ == '__main__':
    args = parse_args()
    main(args)
