import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import Input, Embedding, Flatten, Multiply, Concatenate, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import surprise
from surprise import KNNBasic, NMF, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict

import pickle

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Classification with Embedding Features")

path = '/'.join([os.path.dirname(os.path.realpath('backend.py')), 'data'])


def load_ratings():
    f_path = '/'.join([path, 'ratings_small.csv'])
    return pd.read_csv(f_path)


def load_course_sims():
    f_path = '/'.join([path, 'sim.csv'])
    return pd.read_csv(f_path)


def load_courses():
    f_path = '/'.join([path, 'course_processed.csv'])
    df = pd.read_csv(f_path)
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    f_path = '/'.join([path, 'courses_bows.csv'])
    return pd.read_csv(f_path)


def load_course_genres():
    f_path = '/'.join([path, 'course_genre.csv'])
    return pd.read_csv(f_path)


def load_user_profiles():
    f_path = '/'.join([path, 'user_profile.csv'])
    return pd.read_csv(f_path)


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        f_path = '/'.join([path, 'ratings_small.csv'])
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv(f_path, index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# # User Profile Similarity Model
def user_profile_similarity_model(enrolled_course_ids):
    r = {}

    course_genres_df = load_course_genres()
    # ratings_df = load_ratings()

    # Create new user profile vector
    new_user_vector = ((course_genres_df.copy().drop('TITLE', axis=1).set_index('COURSE_ID').loc[
        enrolled_course_ids]) * 3).sum().to_numpy()

    # get the unknown course ids for the current user id
    all_courses = set(course_genres_df['COURSE_ID'])
    unknown_courses = all_courses.difference(enrolled_course_ids)
    unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
    unknown_course_ids = unknown_course_df['COURSE_ID'].values

    # user np.dot() to get the recommendation scores for each course
    recommendation_scores = np.dot(unknown_course_df.iloc[:, 2:].values, new_user_vector)

    for i, c in enumerate(unknown_course_ids):
        r[c] = (recommendation_scores[i] / max(recommendation_scores))

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}

    return r


# # KMeans clustering
def train_cluster(user_profiles, course_genres, enrolled_courses_ids, pca: bool = False, exp_var: int = 80,
                  cluster_no: int = 15,
                  init: str = 'k-means++', n_init: int = 10, tol: float = 0.0001, max_iter: int = 200):
    """
    Returns a fit KMeans model and new user profile matrix with the active user added as last position
        Parameters
            user_profiles       :   weighted user genre preferences matrix
            course_genres       :   sparse matrix with course ids and their genres
            rated_courses       :   courses already seen (and rated) by active user
            cluster_no, optional:   number of clusters, default = 15
            init, optional      :   clustering algorithm, default = k-means++
            n_init, optional    :   number of centroid initializations, default = 10
            tol, optional       :   tolerance of inertia change before declaring convergence, default = 0.0001
            max_iter, optional  :   iterations of algorithm, default = 200
        Returns
            Trained KMeans model
    """

    # Generating active user profile matrix:
    # - get enrolled courses from genre matrix
    # - multiply selected courses' genres with 3 to get the weighted genre matrix
    # - flatten (sum up by column) and store as numpy array
    # - store new user profile vector as 'active_user'
    new_user_vector = ((course_genres.copy().drop('TITLE', axis=1).set_index('COURSE_ID').loc[
        enrolled_courses_ids]) * 3).sum().to_numpy()

    # User profiles dataframe without 'user' column
    up_df = user_profiles.iloc[:, 1:]

    # Generating dataset to fit KMeans to
    # - concat active_user with user_profiles
    # - extract features (genre weights)
    # - standard scale data

    up_df.loc[len(up_df)] = new_user_vector
    features = up_df
    if pca:
        features = do_pca(up_df, exp_var=exp_var)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Clustering training
    # - initialize KMeans according to parameters
    # - fit data to KMeans
    # - return the trained model
    params = {
        'n_clusters': cluster_no,
        'init': init,
        'n_init': n_init,
        'tol': tol,
        'max_iter': max_iter
    }
    km = KMeans(**params)
    km.fit(features_scaled)

    return km


def do_pca(data, exp_var: int = 85):
    """
    Returns a PCA reduced dataset according to requested explained variance.
        Parameters
            data    :   dataset to be reduced
            exp_var :   variance to be explained in percent. Default: 80
        Returns
            DataFrame : DataFrame with columns USER and PC1-n
    """
    # Get component number according to explained variance requirement
    # - modify data: it still contains the "user" column
    # - initialize PCA with increasing number of components
    # - fit and transform the data
    # - check explained variance
    # - if it is equal to or exceeds required explained variance, stop.
    exp_var = exp_var / 100
    n_com = 0
    for i in range(min(data.shape[0], data.shape[1])):
        n_com = i
        pca = PCA(n_components=n_com)
        data_reduced = pca.fit_transform(data)
        if sum(pca.explained_variance_ratio_) >= exp_var:
            break

    # Return reduced data
    # - store reduced data in a dataframe (df)
    # - label column headers with PC0-n
    df = pd.DataFrame(data_reduced).rename({i: 'PC' + str(i + 1) for i in range(n_com)}, axis=1)
    return df


def clustering(model, user_profiles, user_course_ratings, enrolled_courses_ids):
    """
    Returns recommendations for the active user as a dictionary (key:value = COURSE_ID:SCORE).
    Recommendation score is calculated as ratio of number of enrollments per course
    divided by maximum number of enrollments for all recommended courses.
    E.g. if, in the list of recommended courses, course A is found x times
    and the maximum number of in enrollments is Y with course B, then the recommendation score for
    course A is x/y and the recommendation score for B is 1.
        Parameters
            model               :   Pretrained KMeans model
            user course_ratings       :   DataFrame of user ratings
            enrolled_courses_ids:   List of active user's enrolled (rated) courses
        Returns
            Dictionary          :   Key: COURSE_ID, Value: Recommendation Score
    """
    # Predict recommendations for active user
    # - generate dict for user_ids and corresponding cluster (userID : cluster label)
    # - get cluster label for active user (new_user_cluster)
    # - extract all users from that cluster except the active user (sim_user)
    # - check similar users' rated courses in that cluster
    # - rank courses
    # - remove courses already rated by the active_user
    # - transform enrollment numbers into 0-100 scoring

    user_cluster_dict = {u: model.labels_[i] for i, u in
                         enumerate(user_profiles.loc[:, user_profiles.columns == 'user'].user)}
    new_user = max(user_cluster_dict.keys())
    new_user_cluster = user_cluster_dict[new_user]
    sim_users = {
        'user': [user for user in user_cluster_dict if
                 (user_cluster_dict[user] == new_user_cluster) & (user != new_user)]}
    sim_users_df = pd.DataFrame.from_dict(sim_users)
    sim_courses_df = pd.DataFrame.merge(sim_users_df, user_course_ratings, on='user')
    sim_courses_df['count'] = [1] * sim_courses_df.shape[0]
    sim_courses_df = (sim_courses_df.groupby(['item'])
                      .agg(enrollments=('count', 'sum'))
                      .sort_values(by='enrollments', ascending=False)
                      .reset_index()
                      )
    sim_courses_df = sim_courses_df[~(sim_courses_df.item.isin(enrolled_courses_ids))]
    sim_courses_df.enrollments = (sim_courses_df.enrollments / sim_courses_df.enrollments.max()) * 100
    r = {sim_courses_df.item.iloc[i]: sim_courses_df.enrollments.iloc[i] for i in range(sim_courses_df.shape[0])}

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}

    return r


# # KNNBasic (surprise)
def knn_predict(k, enrolled_course_ids):
    """
    Returns predicted course ratings for active user's unseen courses, KNNBasic model trained on user ratings.
        Parameters
            k           :   Number of neighbors to consider (default: 40)
            enrolled_courses_ids:   List of previously rated courses by the active user
        Returns
            Dictionary  :   Key: Value are Course_ID: Predicted Rating

        Note: Switching to user_based = True can take a long time!
        There are tens of thousands of users, so building the similarity matrix is computationally heavy!
    """
    # Get data
    ratings_df = load_ratings()
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))
    data = Dataset.load_from_df(ratings_df, reader)

    # Build a trainset
    trainset = data.build_full_trainset()

    # Setup KNNBasic parameters
    sim_options = {
        'user_based': False
    }
    params = {
        'min_k': 1,
        'sim_options': sim_options,
        'verbose': False
    }

    model = KNNBasic(k, **params)

    # Fit the model
    model.fit(trainset)
    # with open('trainset_knn.pickle', 'rb') as handle:
    #    trainset_knn = pickle.load(handle)

    predictions = model.test(trainset.build_anti_testset())

    new_user_id = ratings_df.user.max()

    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if uid == new_user_id:
            top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

    # Loading needed data
    # - User Ratings
    # - Course IDs
    # - Active user id (user_id, was appended earlier by add_new_ratings())

    r = {}
    for course_id, rates in user_ratings:
        r[course_id] = rates / 3 * 100

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}

    return r


# NMF Model
def nmf_train(n_factors: int = 15, init_low=0.1, verbose: bool = False):
    """
    Returns dictionary of predicted rating scores based on the NMF method
        Parameters
            enrolled_courses    :   List of courses previously rated by the active user

        Returns
            Dictionary  :   Key: Value are Course_ID: Predicted Score
    """

    # load data
    r_df = load_ratings()

    # Build user interaction matrix
    # uim = r_df.pivot(index='user', columns='item', values='rating').fillna(0).to_numpy()
    reader = Reader(line_format='user item rating', rating_scale=(2, 3))
    data = Dataset.load_from_df(r_df, reader=reader)

    trainset = data.build_full_trainset()

    nmf = NMF(n_factors=n_factors, init_low=init_low, verbose=verbose)
    nmf.fit(trainset)

    trainset_path = '/'.join([path, 'trainset.pickle'])
    with open(trainset_path, 'wb') as handle:
        pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return nmf


def nmf_predict(model, user_id, n, verbose: bool = False) -> dict:
    # n - top_courses

    ratings_small_df = load_ratings()

    rating_sparse_df = ratings_small_df.pivot(index='user', columns='item', values='rating').fillna(
        0).reset_index().rename_axis(index=None, columns=None)

    all_courses = rating_sparse_df.copy().set_index('user').columns.to_list()

    rating_sparse_df_copy = rating_sparse_df.copy().set_index('user')

    list_of_unrated_courses = np.nonzero(rating_sparse_df_copy
                                         .loc[user_id]
                                         .to_numpy() == 0)[0]

    user_set = [[user_id, item_id, 0] for item_id in list_of_unrated_courses]

    trainset_path = '/'.join([path, 'trainset.pickle'])
    with open(trainset_path, 'rb') as handle:
        trainset = pickle.load(handle)

    predictions = model.test(user_set)

    # 1. First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # 2. Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        # user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[: n]

    top_n_user = top_n[user_id]

    top_n_user_course_id = []
    for row in top_n_user:
        cid = all_courses[row[0]]
        top_n_user_course_id.append((cid, row[1]))

    # sort dictionary
    r = {k: v * 100 / 3 for k, v in sorted(top_n_user_course_id, key=lambda item: item[1], reverse=True)}

    return r


# NN Model
class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

    def call(self, inputs):
        """
           method to be called during model fitting

           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability

        return tf.nn.relu(x)


def process_dataset(raw_data):
    """Takes ratings dataframe and returns it with user and items number encoded.
    Parameters
            df  :   Pandas Dataframe (long) with user ids, items and ratings (shape: (n, 3))
    Returns
            df  :   Pandas Dataframe (long) with manipulations performed as mentioned above.

            """
    encoded_data = raw_data.copy()

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict


def generate_train_test_datasets(dataset, train_size=0.8, scale=True):
    """ Generate training and validation datasets with interactions binarized ([0, 2, 3] --> [0, 1])"""

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    val_size = 0.9 - train_size
    train_indices = int(train_size * dataset.shape[0])
    test_indices = int((train_size + val_size) * dataset.shape[0])

    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices],
        x[train_indices:test_indices],
        x[test_indices:],
        y[:train_indices],
        y[train_indices:test_indices],
        y[test_indices:],
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def nn_train(train_size=0.8, embedding_size=32, batch_size=64, n_epochs=7):
    # # Generate training dataset
    # THIS IS DONE BY load trained model IN THE def train() now
    # TO ALLOW FOR USER INPUT
    df_ratings = load_ratings()
    encoded_df, user_idx2id_dict, course_idx2id_dict = process_dataset(df_ratings)
    x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_df, train_size)

    # Create and compile the model

    n_users = len(df_ratings['user'].unique()) + 100
    n_items = len(df_ratings['item'].unique())

    nn_model = RecommenderNet(num_users=n_users, num_items=n_items, embedding_size=embedding_size)
    nn_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])

    nn_model._name = 'neural_network_model'
    # Fit the model
    nn_model_hist = nn_model.fit(x_train,
                                 y_train,
                                 validation_split=None,
                                 epochs=n_epochs,
                                 batch_size=batch_size,
                                 verbose=1
                                 )
    return nn_model, nn_model_hist


def nn_predict(model, enrolled_course_ids) -> dict:
    # Get dataset for which to predict
    # - get active user (last user in the ratings dataframe)
    # - extract data only for them
    # - set user_id to 0 for that user
    df_ratings = load_ratings()
    active_user_indx = len(df_ratings['user'].unique()) - 1  # active user index, it will be used to make predictions

    encoded_data, _, course_idx2id_dict = process_dataset(df_ratings)

    # encode user rated courses id
    active_key_courses = []
    for k, v in course_idx2id_dict.items():
        if v in enrolled_course_ids:
            active_key_courses.append(k)
        else:
            continue

    # get encoded unseen courses for active user
    active_user_unseen_courses = set(encoded_data['item']) - set(active_key_courses)

    # get dataframe for use in the model
    active_user_array = np.array([active_user_indx] * len(active_user_unseen_courses))
    df_active_user_to_predict = pd.DataFrame(list(zip(active_user_array,
                                                      active_user_unseen_courses)),
                                             columns=['user', 'item'])

    # make the model prediction
    active_user_predictions = model.predict(df_active_user_to_predict)

    # Store predictions in new column and scale scores to be between 0 and 100 (now: 0, 1)
    df_active_user_to_predict['recommend_score'] = active_user_predictions * 100

    # return recommendations for the user as a 'sorted dictionary' of structure: COURSE_ID : RECOMMENDATION_SCORE
    # - re-transform item_ids from numbers to course ids
    # - use course id and recommend_score columns and set course id as the index
    # - sort the dictionary and return
    df_active_user_to_predict['COURSE_ID'] = df_active_user_to_predict['item'].map(course_idx2id_dict)
    df_active_user_to_predict = df_active_user_to_predict[['COURSE_ID', 'recommend_score']].set_index('COURSE_ID')
    r = {}
    r = df_active_user_to_predict.to_dict()

    r = {k: v for k, v in sorted(r['recommend_score'].items(), key=lambda item: item[1], reverse=True)}

    return r


# Embeddings with Classification
def get_embeddings():
    # Load pretrained model
    model_path = '/'.join([path, 'saved_model/nn_model_emb'])
    nn_model_emb = tf.keras.models.load_model(model_path)

    user_embedding = nn_model_emb.get_layer('user_embedding_layer').get_weights()[0]
    item_embedding = nn_model_emb.get_layer('item_embedding_layer').get_weights()[0]

    return user_embedding, item_embedding


def emb_df_create():
    # get user and item embeddings
    user_embedding, item_embedding = get_embeddings()

    # get data
    df_ratings = load_ratings()
    df_pivoted = (df_ratings.pivot(index='user', columns='item', values='rating')
                  .fillna(0).reset_index().rename_axis(index=None, columns=None))

    # create user-embedding df with codding column names by user feature numbers
    user_emb_df = pd.DataFrame(user_embedding)
    user_emb_df = user_emb_df.merge(df_pivoted['user'], left_index=True, right_index=True)
    user_emb_df.columns = [f'UFeature{i}' for i in range(32)] + ['user']

    # create item-embedding df with codding column names by numbers
    item_emb_df = pd.DataFrame(item_embedding)
    courses_ids = df_pivoted.keys()[1:]
    course_id2idx_dict = {x: i for i, x in enumerate(courses_ids)}
    item_emb_df['courses'] = courses_ids
    item_emb_df.columns = [f'CFeature{i}' for i in range(32)] + ['item']

    # Merge user embedding features
    user_emb_merged = pd.merge(df_ratings, user_emb_df, how='left', left_on='user', right_on='user').fillna(0)
    # Merge course embedding features
    merged_df = pd.merge(user_emb_merged, item_emb_df, how='left', left_on='item', right_on='item').fillna(0)

    u_features = [f"UFeature{i}" for i in range(32)]
    c_features = [f"CFeature{i}" for i in range(32)]

    # Extract user embedding features
    user_embeddings = merged_df[u_features]
    # Extract course embedding features
    course_embeddings = merged_df[c_features]
    # Extract ratings
    ratings = merged_df['rating']

    # Aggregate the two feature columns using element-wise add
    regression_df = user_embeddings + course_embeddings.values
    # Rename the columns of the resulting DataFrame
    regression_df.columns = [f"Feature{i}" for i in range(32)]
    # Add the 'rating' column from the original DataFrame to the regression dataset
    regression_df['rating'] = ratings

    return regression_df


def emb_classification_train(regression_df, n_neighbors=20):
    # take ready df prepared before from embedding vectors and treat it as classification problem with KNN

    X = regression_df.iloc[:, :-1]
    y = regression_df.iloc[:, -1]

    # Encode y column with ratings 3 and 2 to binary format with 0 and 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.values.ravel())

    # Train the model
    knn_emb = KNeighborsClassifier(n_neighbors, n_jobs=-1)
    knn_emb.fit(X, y)

    return knn_emb


def emb_classification_predict(model, enrolled_course_ids):
    """
        Predicts neighbors for the latest user based on the embeddings generated earlier.
        Returns a dictionary in the form of COURSE_ID: SCORE
    """
    df_ratings = load_ratings()
    regression_df = emb_df_create()
    print('active user', regression_df.iloc[-1, :-1].values.reshape(1, -1))
    knn_emb_pred = knn_emb.kneighbors(regression_df.iloc[-1, :-1].values.reshape(1, -1), n_neighbors=10)

    # get nearest neighbors indexes for latest user
    knn_pred_df = pd.DataFrame(knn_emb_pred[1].reshape(10, 1)).rename({0: 'UserIdx'}, axis=1)
    print('Predictions', knn_pred_df)

    knn_courses = list()
    for u in knn_pred_df['UserIdx'].values:
        # print(u)
        for c in df_ratings[df_ratings['user'] == df_ratings.iloc[u]['user']]['item']:
            # print(c)
            if c not in enrolled_course_ids:
                knn_courses.append(c)
    # print(knn_courses)

    max_rate = pd.Series(knn_courses, dtype=object).value_counts().max()
    # print(max_rate)
    r = pd.Series(knn_courses, dtype=object).value_counts().apply(lambda x: x / max_rate * 100).to_dict()
    # print(r)

    return r


# Returned models:
km = KMeans()  # untrained KMeans model
knn = KNNBasic()  # untrained KNNBasic model
nmf = NMF()  # untrained NMF model
nn = None  # untrained Neural Network
knn_emb = None  # untrained embeddings classifier


# Model training
def train(model_name, enrolled_course_ids, params):
    global km, knn, nmf, nn, knn_emb
    if model_name in [models[0], models[1]]:
        pass
    if model_name in [models[2], models[3]]:
        km = train_cluster(user_profiles=load_user_profiles(),
                           course_genres=load_course_genres(),
                           enrolled_courses_ids=enrolled_course_ids,
                           pca=(model_name == models[3]),
                           exp_var=params['exp_var'],
                           cluster_no=params['cluster_no']
                           )
    if model_name == models[4]:
        pass

    if model_name == models[5]:
        nmf = nmf_train(n_factors=params['nmf_factors'], init_low=0.1,
                        verbose=False)
    if model_name == models[6]:
        # Check whether to use default or not
        if ((params['nn_train_size'] == 0.8) & (params['nn_batch_size'] == 64)
                & (params['nn_epochs'] == 7) & (params['nn_embedding_size'] == 32)):
            f_path = '/'.join([path, 'saved_model/nn_model'])
            nn = tf.keras.models.load_model(f_path)
        else:
            train_size = params['nn_train_size']
            n_epochs = params['nn_epochs']
            batch_size = params['nn_batch_size']
            embedding_size = params['nn_embedding_size']
            nn, _ = nn_train(train_size, embedding_size, batch_size, n_epochs)

    if model_name == models[7]:
        regression_df = emb_df_create()
        n_neighbors = params['emb_n_neighbors']
        knn_emb = emb_classification_train(regression_df, n_neighbors)


# Prediction
def predict(model_name, user_id, params):
    ratings_df = load_ratings()
    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].to_list()
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    # Course Similarity model
    if model_name == models[0]:
        top_courses = params['top_courses']
        sim_threshold = 0.6
        if "sim_threshold" in params:
            sim_threshold = params["sim_threshold"] / 100.0
        res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
        for key, score in res.items():
            if score >= sim_threshold:
                users.append(user_id)
                courses.append(key)
                scores.append(score)

    if model_name == models[1]:
        top_courses = params['top_courses']
        sim_threshold = params['sim_threshold'] / 100.0
        res = user_profile_similarity_model(enrolled_course_ids)
        for key, score in res.items():
            if score >= sim_threshold:
                courses.append(key)
                scores.append(score)

    if model_name in [models[2], models[3]]:
        top_courses = params['top_courses']
        sim_threshold = params['km_sim_threshold']
        res = clustering(km, load_user_profiles(), load_ratings(), enrolled_course_ids)
        for key, score in res.items():
            if score >= sim_threshold:
                users.append(user_id)
                courses.append(key)
                scores.append(score)

    # k-Nearest Neighbor (KNNBasic, collaborative filtering)
    if model_name == models[4]:
        top_courses = params['top_courses']
        sim_threshold = params['knn_sim_threshold']
        k = params['n_neigh']
        res = knn_predict(k, enrolled_course_ids)
        for key, score in res.items():
            if score >= sim_threshold:
                courses.append(key)
                scores.append(score)

    # NMF Model (collaborative filtering)
    if model_name == models[5]:
        top_courses = params['top_courses']
        sim_threshold = params['nmf_sim_threshold']
        res = nmf_predict(nmf, user_id, top_courses, False)
        for key, score in res.items():
            if score * 100 / 3 >= sim_threshold:
                courses.append(key)
                scores.append(score)

    # Neural Network Model (collaborative filtering)
    if model_name == models[6]:
        top_courses = params['top_courses']
        sim_threshold = params['nn_sim_threshold']
        res = nn_predict(nn, enrolled_course_ids)
        for key, score in res.items():
            if score >= sim_threshold:
                courses.append(key)
                scores.append(score)

    # Classification-based Model using Embedding Features ((collaborative filtering)
    if model_name == models[7]:
        top_courses = params['top_courses']
        sim_threshold = params['emb_sim_threshold']
        res = emb_classification_predict(knn_emb, enrolled_course_ids)
        for key, score in res.items():
            if score >= sim_threshold:
                courses.append(key)
                scores.append(score)

    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['COURSE_ID', 'SCORE'])

    return res_df
