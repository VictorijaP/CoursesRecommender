import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache
def load_ratings():
    return backend.load_ratings()

@st.cache
def load_course_sims():
    return backend.load_course_sims()


@st.cache
def load_courses():
    return backend.load_courses()


@st.cache
def load_bow():
    return backend.load_bow()

@st.cache
def load_course_genres():
    return backend.load_course_genres()

@st.cache
def load_user_profiles():
    return backend.load_user_profiles()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, enrolled_course_ids, params):

    with st.spinner('Training...'):
        backend.train(model_name, selected_courses_df['COURSE_ID'].values, params)
        st.success('Done!')

def predict(model_name, new_user_id, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, new_user_id, params)
    st.success('Recommendations generated!')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=20,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
# User profile model
elif model_selection == backend.models[1]:
    profile_sim_threshold = st.sidebar.slider('User Profile Similarity Threshold %',
                                              min_value=0, max_value=100,
                                              value=50, step=10)
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=20,
                                    value=10, step=1)
    params['top_courses'] = top_courses
    params['sim_threshold'] = profile_sim_threshold
# Clustering model
elif model_selection == backend.models[2]:
    km_sim_threshold = st.sidebar.slider('Clustering Similarity Threshold %',
                                      min_value=0, max_value=100,
                                      value=50, step=10)
    cluster_no = st.sidebar.slider('Number of Clusters', min_value=5, max_value=50, value=20, step=1)
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    params['km_sim_threshold'] = km_sim_threshold
    params['top_courses'] = top_courses
    params['cluster_no'] = cluster_no
    params['exp_var'] = 85
# Clustering + PCA model
elif model_selection == backend.models[3]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=20, value=10, step=1)
    n_clu_pca = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    exp_var = st.sidebar.slider('Explained Variance', min_value=1, max_value=100, value=80, step=1)
    pca_sim_threshold = st.sidebar.slider('PCA Clustering Similarity Threshold %',
                                      min_value=0, max_value=100,
                                      value=50, step=10)
    params['top_courses'] = top_courses
    params['cluster_no'] = n_clu_pca
    params['exp_var'] = exp_var
    params['km_sim_threshold'] = pca_sim_threshold
# KNN
elif model_selection == backend.models[4]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=20, value=10, step=1)
    n_neigh = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=50, value=40, step=1)
    knn_sim_threshold = st.sidebar.slider('KNN Similarity Threshold %',
                                          min_value=0, max_value=100,
                                          value=50, step=10)
    params['top_courses'] = top_courses
    params['n_neigh'] = n_neigh
    params['knn_sim_threshold'] = knn_sim_threshold
# NMF model
elif model_selection == backend.models[5]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    nmf_factors = st.sidebar.slider('NMF Factors', min_value=1, max_value=30, value=15, step=1)
    nmf_epochs = st.sidebar.slider('SGD Epochs', min_value=1, max_value=100, value=50, step=1)
    nmf_sim_threshold = st.sidebar.slider('NMF Similarity Threshold %',
                                      min_value=0, max_value=100,
                                      value=50, step=10)
    params['top_courses'] = top_courses
    params['nmf_factors'] = nmf_factors
    params['nmf_epochs'] = nmf_epochs
    params['nmf_sim_threshold'] = nmf_sim_threshold
# Neural Net
elif model_selection == backend.models[6]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=20, value=10, step=1)
    nn_sim_threshold = st.sidebar.slider('NN Similarity Threshold %',
                                          min_value=0, max_value=100,
                                          value=50, step=10)
    params['nn_train_size'] = 0.8
    params['nn_batch_size'] = 64
    params['nn_epochs'] = 7
    params['nn_embedding_size'] = 32
    params['nn_sim_threshold'] = nn_sim_threshold
    with st.sidebar.expander('Advanced options.', False):
        st.caption('Changing these options will cause the app to train a new model, instead of using the pretrained one.')
        st.info('Depending on your settings, training can take a significant amount of time (minutes to many hours).')
        st.caption('Training dataset generation')
        nn_train_size = st.slider('Training data size', min_value=0.5, max_value=0.9, value=0.8, step=0.05)
        nn_batch_size = st.slider('Batch size', min_value=16, max_value=512, value=64, step=16)
        st.caption('Model training')
        nn_epochs = st.slider('Epochs', min_value=1, max_value=30, value=7, step=1)
        nn_embedding_size = st.slider('Embedding_size', min_value=8, max_value=96, value=32, step=8)
    params['nn_train_size'] = nn_train_size
    params['nn_batch_size'] = nn_batch_size
    params['nn_epochs'] = nn_epochs
    params['nn_embedding_size'] = nn_embedding_size
    params['top_courses'] = top_courses
    params['nn_sim_threshold'] = nn_sim_threshold
# Classification with embeddinds model
elif model_selection == backend.models[7]:
    top_courses = st.sidebar.slider('Top courses', min_value=1, max_value=20, value=10, step=1)
    emb_sim_threshold = st.sidebar.slider('Classification Similarity Threshold %',
                                         min_value=0, max_value=100,
                                         value=50, step=10)
    emb_n_neighbors = st.sidebar.slider('Niarest Neighbors', min_value=1, max_value=50,
                                         value=20, step=1)
    params['top_courses'] = top_courses
    params['emb_sim_threshold'] = emb_sim_threshold
    params['emb_n_neighbors'] = emb_n_neighbors

else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, selected_courses_df['COURSE_ID'].values, params)


# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_user_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    #user_ids = [new_id]
    res_df = predict(model_selection, new_user_id, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
    res_df = res_df[['TITLE', 'DESCRIPTION', 'SCORE']].head(params['top_courses'])
    st.table(res_df)
