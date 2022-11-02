
'''
Codigo para la creacion de una aplicacion para el uso del VRA en streamlit
'''

# %%
# Se cargan las librerías necesarias para realizar este proceso

import pandas as pd
import boto3
from botocore.exceptions import ClientError
import ast
from scipy import spatial
import numpy as np
import streamlit as st

# %%
# Se cargan las claves necesarias para utilizar a lo largo del proceso
# También se necesitan claves de acceso a nuestro servidor de AWS

BUCKET_S3 = st.secrets['bucket_s3']
NEW_FILE_NAME = 'clean_dataset/games_clean.feather'
COMPLEX_NAME = 'clean_dataset/games_complex.feather'
CLEAN_FOLDER = 'clean_reviews/'
COLS_INT = [
    'age_ratings', 'OC_rating', 'first_release_date', 'RAWG_nreviews'
    ]
COLS_LIST = [
    'porting', 'supporting', 'platforms', 'series', 'game_modes', 'genres',
    'player_perspectives', 'themes', 'developer', 'publisher', 'devs',
    'keywords', 'franchises', 'country'
    ]
COLS_FLOAT = [
    'main_duration', 'extra_duration', 'comp_duration', 'RAWG_rating'
    ]
DURATION_DICT = {
    'Only main content': 'main_duration',
    'Extra content': 'extra_duration',
    'Completionist': 'comp_duration'
    }

one_hot_values = {
    'developer': 0.75,
    'publisher': 0.25,
    'keywords': 0.6,
    'devs': 1,
    'franchises': 1,
    'country': 0.05,
    'genres': 0.9,
    'themes': 0.7,
    'game_modes': 0.5,
    'player_perspectives': 0.4
    }

MULTI_COLS = {
    'platforms': 'Select your platforms',
    'genres': 'Select your favorite genres', 
    'themes': 'Select your favorite themes',
    'game_modes': 'Select how you want to play',
    'player_perspectives': 'Select your favorite perspectives'
    }

# %%
# Se leen las reviews disponibles

if 'reviews_df' not in st.session_state:
    bucket = (
        boto3.resource('s3', region_name='us-east-1')
        .Bucket(name=BUCKET_S3[5:])
        )
    av_files = [
        obj.key for obj in bucket.objects.filter(Prefix=CLEAN_FOLDER)
        if len(obj.key) > len(CLEAN_FOLDER)
        ]
    
    reviews_list = []
    for file in av_files:
        reviews_list.append(
            pd.read_feather(f'{BUCKET_S3}/{file}')
            )
    
    st.session_state.reviews_df = pd.concat(reviews_list).drop('id', axis=1)
    print('Reviews cargadas')

# %%
# Se leen los archivos de juegos
if 'games_df' not in st.session_state:
    games_df = (
        pd.read_feather(f'{BUCKET_S3}/{NEW_FILE_NAME}')
        )
    
    complex_df = (
        pd.read_feather(f'{BUCKET_S3}/{COMPLEX_NAME}')
        )

    complex_df['storyline'] = (
        complex_df['storyline'].replace('nan', '')
        )
    for col in COLS_LIST[:4]:
        complex_df[col] = complex_df[col].map(ast.literal_eval)
    for col in COLS_LIST[4:11]:
        games_df[col] = games_df[col].map(ast.literal_eval)
        complex_df[col] = complex_df[col].map(ast.literal_eval)
    for col in COLS_LIST[11:]:
        games_df[col] = games_df[col].map(ast.literal_eval)
    
    games_df['OC_rating'] = games_df['OC_rating'].str[:-2]
    complex_df['OC_rating'] = complex_df['OC_rating'].str[:-2]
    complex_df['age_ratings'] = complex_df['age_ratings'].str[:-2]
        
    for col in COLS_FLOAT[:-1]:
        games_df[col] = games_df[col].astype(float)
        complex_df[col] = complex_df[col].astype(float)
    
    games_df[COLS_INT[1:]] = games_df[COLS_INT[1:]].astype(int)
    complex_df[COLS_INT[:1]] = complex_df[COLS_INT[:1]].astype(int)
    games_df[COLS_FLOAT] = games_df[COLS_FLOAT].astype(float)
    complex_df[COLS_FLOAT[:-1]] = complex_df[COLS_FLOAT[:-1]].astype(float)
    complex_df['first_release_date'] = pd.to_datetime(
        complex_df['first_release_date']
        ).dt.date

    st.session_state.games_df = games_df.copy()
    st.session_state.complex_df = complex_df.copy()

# %%
# Se selecciona un juego

with st.form(key='select_filters'):
    game_name = st.selectbox(
        'Select a game',
        [''] + st.session_state.complex_df['name'].tolist(),
        key='game_1'
        )
    
    # Se pide seleccionar las plataformas en las que se quiere jugar
    available = []
    selected = []
    for col in MULTI_COLS.keys():
        available.append(
            st.session_state
            .complex_df[col]
            .explode()
            .dropna()
            .sort_values()
            .unique()
            )
        selected.append(
            st.multiselect(
                MULTI_COLS[col],
                available[-1],
                help=(
                    'Selecting none will mean that no filter is applied'
                    )
                )
            )
    
    if 'PlayStation 5' in selected[0]:
        selected[0].append('PlayStation 4')
    if 'PlayStation 2' in selected[0]:
        selected[0].append('PlayStation')
    if 'Xbox Series X|S' in selected[0]:
        selected[0] += ['Xbox One', 'Xbox 360']
    if 'Wii U' in selected[0]:
        selected[0].append('Wii')
    if 'Nintendo DSi' in selected[0]:
        selected[0].append('Nintendo DS')
    if 'Nintendo 3DS' in selected[0]:
        selected[0] += ['Nintendo DS', 'Nintendo DSi']
    if 'New Nintendo 3DS' in selected[0]:
        selected[0] += ['Nintendo DS', 'Nintendo DSi', 'Nintendo 3DS']
    if 'Game Boy Color' in selected[0]:
        selected[0].append('Game Boy')
    if 'Game Boy Advance' in selected[0]:
        selected[0] += ['Game Boy', 'Game Boy Color']
        
    if selected[0] == []:
        selected[0] = list(available[0])
    selected[0] = set(selected[0])

    with st.expander("Select a duration if you'd like"):
        duration_type = DURATION_DICT[
                st.radio(
                'Select what type of player you are',
                DURATION_DICT.keys(),
                )
                ]
        game_duration = list(
                st.slider(
                'Select your desired duration',
                value=(0,100),
                help='Selecting 100 means you are choosing 100+ hours games'
                )
                )

    with st.expander("Select an age limit if you'd like"):
        age = st.radio(
            'Selecciona un limite de edad',
            st.session_state.complex_df['age_ratings'].sort_values().unique(),
            index=4
            )

    same_dev = st.checkbox(
        "Leave this checked if you'd like results from the same developers as"
        " the game you selected",
        True
        )

    fran = st.checkbox(
        "Leave this checked if you'd like results from the same game series as"
        " the game you selected",
        True
        )
    
    SEND_GAME = st.form_submit_button('Buscar recomendaciones')

if SEND_GAME:
    if game_name == '':
        st.error("You didn't select any games")
    else:
        with st.spinner('Searching for diamonds in the mine'):
            complex_df = st.session_state.complex_df.copy()
            if not same_dev:
                game_dev = st.session_state.complex_df.loc[
                    st.session_state.complex_df['name'] == game_name,
                    'developer'
                    ].iloc[0]
                if len(game_dev) > 0:
                    complex_df = (
                        complex_df
                        .loc[~(
                            complex_df['developer'].map(
                                lambda x: any(
                                    [True for ser in x if ser in game_dev]
                                    )
                                )
                            )
                            ]
                        )
            if not fran:
                game_ser = st.session_state.complex_df.loc[
                    st.session_state.complex_df['name'] == game_name, 'series'
                    ].iloc[0]
                if len(game_ser) > 0:
                    complex_df = (
                        complex_df
                        .loc[~(
                            complex_df['series'].map(
                                lambda x: any(
                                    [True for ser in x if ser in game_ser]
                                    )
                                )
                            )
                            ]
                        )
    
            for var, col in zip(
                    selected,
                    MULTI_COLS.keys()
                    ):
                if len(var) > 0:
                    complex_df = (
                        complex_df
                        .loc[
                            complex_df[col].map(
                                lambda x: any(
                                    [True for gen in x if gen in var]
                                    )
                                )
                            ]
                        )
            
            if game_duration[1] == 100:
                game_duration[1] = 100000
            complex_df = complex_df.loc[
                (complex_df[duration_type] >= game_duration[0]) &
                (complex_df[duration_type] <= game_duration[1])
                ]
    
            complex_df = complex_df.loc[complex_df['age_ratings'] <= age]
    
            if len(complex_df) == 0:
                st.warning("There aren't any games that match those filters")
            else:
                complex_df = (
                    pd.concat([
                        complex_df,
                        st.session_state.complex_df.loc[
                            st.session_state.complex_df['name'] == game_name
                            ]
                        ])
                    .drop_duplicates('name')
                    )
                games_df = st.session_state.games_df.merge(
                    complex_df[['name']], on='name'
                    )
                similar_df = (
                    games_df
                    .drop([
                        'game_id', 'RAWG_rating', 'RAWG_nreviews'
                        ],
                        axis=1
                        )
                    )
                # %%
                # Definicion de las funciones
                
                
                def similarity(name1, name2):
                    '''
                    Funcion que describira la distancia entre dos titulos
                    diferentes
                    '''
                    game1 = similar_df.loc[similar_df['name'] == name1].iloc[0]
                    game2 = similar_df.loc[similar_df['name'] == name2].iloc[0]
                
                    distance = 0
                    for col in similar_df.columns[6:]:
                        # print(col)
                        if any([max(game1[col]) == 0, max(game2[col]) == 0]):
                            add = 1
                        else:
                            add = spatial.distance.cosine(
                                game1[col], game2[col]
                                )
                        add = add * one_hot_values[col]
                        # print(add)
                        distance += add
                    add_year = (
                        (game1['first_release_date'] -
                         game2['first_release_date'])/100
                        )
                    # print(add_year)
                    distance += add_year
                    add_OC = (game1['OC_rating'] - game2['OC_rating'])/30
                    # print(add_OC)
                    distance += add_OC
                    add_duration = np.array([
                        abs(game1['main_duration'] - game2['main_duration'])
                        / 30,
                        abs(game1['extra_duration'] - game2['extra_duration'])
                        / 30,
                        abs(game1['comp_duration'] - game2['comp_duration'])
                        / 30
                        ]).mean()
                    if add_duration > 4:
                        add_duration = 4
                    # print(add_duration)
                    distance += add_duration
                    return distance


                # Definicion de la funcion que determina el juego mas similar a
                # uno dado
                
                
                def most_similar(name):
                    '''
                    Funcion que obtendra los n juegos mas cercanos al provisto
                    '''
                    game = similar_df.loc[similar_df['name'] == name].iloc[0]
                    games_to_analize = (
                        similar_df.loc[similar_df['name'] != name].reset_index(
                            drop=True
                            )
                        )
                    games_to_analize['sim'] = (
                        games_to_analize.apply(
                            lambda row: similarity(name, row['name']),
                            axis=1
                            )
                        )
                    games_to_analize = games_to_analize.sort_values(
                        'sim', ascending=True
                        )
                    return games_to_analize
                
                
                def similar_review(game_name):
                    '''
                    Se buscan los juegos que hayan jugado otros usuarios que
                    hayan jugado a dicho juego
                    '''
                    # Se obtiene el id y el numero de reviews del juego
                    game_id = (
                        games_df
                        .loc[games_df['name'] == game_name, 'game_id']
                        .iloc[0]
                        )
                    # Se obtienen los usuarios que hayan jugado al juego
                    game_df = (
                        pivoted_df
                        .dropna(subset=game_id)
                        .loc[pivoted_df[game_id] > 0]
                        )
                    # Se obtiene la columna del juego solicitado
                    game_col = game_df[game_id]
                    # Se obtienen solo los juegos que hayan jugado un % de los
                    # jugadores
                    games_played = (
                        game_df
                        .drop(game_id, axis=1)
                        .count()
                        .loc[lambda value:
                             (value >= len(game_col) * 0.1) & (value >= 5)
                             ]
                        .index
                        )
                    games_played_df = game_df[games_played].fillna(0)
                
                    # Se correlan los valores y se ordena
                    return (
                        games_played_df
                        .corrwith(game_col)
                        .sort_values(ascending=False)
                        .loc[lambda value: value > 0]
                        .reset_index()
                        .rename(columns={0: 'corr'})
                        .merge(games_df[['name', 'game_id']], on='game_id')
                        [['name', 'corr']]
                        )
                
                
                def user_based_recommender(
                        user_slug, perc=50, n_users=10
                        ):
                    if perc > 1:
                        perc = perc/100
                    elif perc <= 0:
                        return 'Pero no ves que no'
                    user_df = pivoted_df.loc[pivoted_df.index == user_slug]
                    user_played_games = (
                        user_df.columns[user_df.notna().any()].tolist()
                        )
                    user_played_df = user_df[user_played_games]
                    
                    pivoted_user_df = pivoted_df[user_played_games]
                    limited_user_df = pivoted_user_df.loc[
                        pivoted_user_df.count(axis=1) >=
                        (len(user_played_games) * perc)
                        ]
                    corr_values = (
                        limited_user_df
                        .T
                        .corr()
                        .loc[limited_user_df.index == user_slug]
                        .drop(user_slug, axis=1)
                        .reset_index(drop=True)
                        .unstack()
                        .sort_values(ascending=False)
                        .reset_index()
                        .drop('level_1', axis=1)
                        .rename(columns={0: 'corr'})
                        .dropna()
                        )
                
                    top_related_users = corr_values.iloc[:n_users]
                    avg_corr = top_related_users['corr'].mean()
                    top_related_rating = (
                        top_related_users.merge(reviews_df)
                        .assign(weighted_rating=lambda df:
                                df['corr'] * df['review_rating'])
                        .drop('user_id', axis=1)
                        )
                
                    non_played_rating = top_related_rating.loc[
                        ~(top_related_rating['game_id']
                          .isin(user_played_games))
                        ]
                    recommendation_df = (
                        non_played_rating
                        # top_related_rating
                        .groupby('game_id')
                        ['weighted_rating']
                        .agg(['count', 'mean'])
                        .reset_index()
                        .loc[lambda df:
                             (df['count'] >= n_users * perc) &
                             (df['mean'] >= 4*avg_corr)
                             ]
                        .sort_values('mean', ascending=False)
                        .drop(['count'], axis=1)
                        .merge(games_df[['name', 'game_id']], on='game_id')
                        [['name', 'mean']]
                        )
                
                    return recommendation_df


                complex_df = complex_df.rename(columns={
                    col: col.title().replace('_', ' ')
                    for col in complex_df.columns
                    })
                complex_df = complex_df.rename(columns={'Oc Rating': 'Rating'})
            
                results = (
                    most_similar(game_name)
                    [['name']]
                    .merge(complex_df, left_on='name', right_on='Name')
                    .drop('name', axis=1)
                    )

                pivoted_df = (
                    games_df
                    .merge(
                        st.session_state.reviews_df, how='left', on='game_id'
                        )
                    .pivot_table(
                        index='user_id',
                        columns='game_id',
                        values='review_rating'
                        )
                    .replace([1, 3, 4, 5], [-2, 0, 1, 5])
                    )

                results_2 = (
                    similar_review(game_name)
                    [['name']]
                    .merge(complex_df, left_on='name', right_on='Name')
                    .drop('name', axis=1)
                    )

                chosen_game = (
                    complex_df
                        .loc[complex_df['Name'] == game_name]
                        .reset_index(drop=True)
                        )

        st.write('This is your chosen game')
        st.dataframe(chosen_game)
        st.write('These are recommended based on similarity among games')
        st.dataframe(results.iloc[:10])
        st.write(
            'These are recommended based on what other players have also '
            'played'
            )
        st.dataframe(results_2.iloc[:10])