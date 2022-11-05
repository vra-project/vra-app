
'''
Codigo para la creacion de una aplicacion para el uso del VRA en streamlit
'''

# %%
# Se cargan las librerias necesarias para realizar este proceso

import random as rd
import ast
import boto3
import pandas as pd
from scipy import spatial
import numpy as np
import streamlit as st

# %%
# Se cargan las claves necesarias para utilizar a lo largo del proceso
# También se necesitan claves de acceso a nuestro servidor de AWS

BUCKET_S3 = st.secrets['bucket_s3']
CLEAN_NAME = st.secrets['clean_name']
COMPLEX_NAME = st.secrets['complex_name']
CLEAN_FOLDER = st.secrets['clean_folder']
COLS_INT = [
    'age_ratings', 'OC_rating', 'first_release_date', 'RAWG_nreviews'
    ]
COLS_LIST = [
    'platforms', 'series', 'game_modes', 'genres', 'player_perspectives',
    'themes', 'developer', 'publisher', 'devs', 'keywords', 'franchises',
    'country'
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

EASTER_EGGS = st.secrets['easter_eggs'].split(', ')

# %%
# Se leen las reviews disponibles
with st.spinner('Loading data'):
    if 'reviews_df' not in st.session_state:
        bucket = (
            boto3.resource('s3', region_name='us-east-1')
            .Bucket(name=BUCKET_S3[5:])
            )
        av_files = [
            obj.key for obj in bucket.objects.filter(
                Prefix=CLEAN_FOLDER+'reviews'
                )
            if len(obj.key) > len(CLEAN_FOLDER)
            ]

        reviews_list = []
        for file in av_files:
            reviews_list.append(
                pd.read_feather(f'{BUCKET_S3}/{file}')
                )

        st.session_state.reviews_df = pd.concat(reviews_list)
        print('Reviews cargadas')

    # %%
    # Se leen los archivos de juegos
    if 'games_df' not in st.session_state:
        games_df = (
            pd.read_feather(f'{BUCKET_S3}/{CLEAN_NAME}')
            )

        complex_df = (
            pd.read_feather(f'{BUCKET_S3}/{COMPLEX_NAME}')
            )

        complex_df['storyline'] = (
            complex_df['storyline'].replace('nan', '')
            )
        for col in COLS_LIST[:2]:
            complex_df[col] = complex_df[col].map(ast.literal_eval)
        for col in COLS_LIST[2:9]:
            games_df[col] = games_df[col].map(ast.literal_eval)
            complex_df[col] = complex_df[col].map(ast.literal_eval)
        for col in COLS_LIST[9:]:
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
st.title('Game Based Recommender')
st.write(
    '''
    Welcome to the Game Based Recommender page from VRA. \n
    Here you will be able to get recommendations based on a game you like. \n
    If you prefer to obtain recommendations based on a your username and your
    reviews, you can access to the User Based Page.
    '''
    )
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
            'Select an age limit',
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

    SEND_GAME = st.form_submit_button('Search recommendations')

if SEND_GAME:
    if game_name == '':
        st.error("You didn't select any games")
    else:
        with st.spinner(rd.choice(EASTER_EGGS)):
            complex_df = st.session_state.complex_df.copy()
            # Se filtran juegos de la misma desarrolladora
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
            # Se filtran juegos de la misma franquicia
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
            # Se realizan varios filtros distintos
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
            # Se filtra por duracion
            if game_duration[1] == 100:
                game_duration[1] = 100000
            complex_df = complex_df.loc[
                (complex_df[duration_type] >= game_duration[0]) &
                (complex_df[duration_type] <= game_duration[1])
                ]
            # Se filtra por edad
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
                        if any([max(game1[col]) == 0, max(game2[col]) == 0]):
                            add = 1
                        else:
                            add = spatial.distance.cosine(
                                game1[col], game2[col]
                                )
                        add = add * one_hot_values[col]
                        distance += add
                    add_year = (
                        (game1['first_release_date'] -
                         game2['first_release_date'])/100
                        )
                    distance += add_year
                    add_oc = (game1['OC_rating'] - game2['OC_rating'])/50
                    distance += add_oc
                    add_duration = np.array([
                        abs(game1['main_duration'] - game2['main_duration'])
                        / 50,
                        abs(game1['extra_duration'] - game2['extra_duration'])
                        / 50,
                        abs(game1['comp_duration'] - game2['comp_duration'])
                        / 50
                        ]).mean()
                    if add_duration > 4:
                        add_duration = 4
                    distance += add_duration
                    return distance


                # Definicion de la funcion que determina el juego mas similar a
                # uno dado


                def most_similar(name):
                    '''
                    Funcion que obtendra los n juegos mas cercanos al provisto
                    '''
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
                if len(results) > 0:
                    results.index = range(1, len(results)+1)

                # Se prepara el dataset para obtener las reviews similares
                pivoted_df = (
                    games_df
                    .loc[games_df['RAWG_nreviews'] > 4]
                    .merge(
                        st.session_state.reviews_df, how='left', on='game_id'
                        )
                    .pivot_table(
                        index='user_id',
                        columns='game_id',
                        values='review_rating'
                        )
                    .replace([1, 3, 4, 5], [-1, 0, 1, 5])
                    )


                def similar_review(game_title):
                    '''
                    Se buscan los juegos que hayan jugado otros usuarios que
                    hayan jugado a dicho juego
                    '''
                    # Se obtiene el id y el numero de reviews del juego
                    game_id = (
                        games_df
                        .loc[games_df['name'] == game_title, 'game_id']
                        .iloc[0]
                        )
                    # Se obtienen los usuarios que hayan jugado al juego
                    if game_id not in pivoted_df:
                        return pd.DataFrame(columns=['name', 'corr'])
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

                results_2 = (
                    similar_review(game_name)
                    [['name']]
                    .merge(complex_df, left_on='name', right_on='Name')
                    .drop('name', axis=1)
                    )
                if len(results_2) > 0:
                    results_2.index = range(1, len(results_2)+1)

                chosen_game = (
                    complex_df
                        .loc[complex_df['Name'] == game_name]
                        .reset_index(drop=True)
                        )
                chosen_game.index = [1]

        st.write('This is your chosen game')
        st.dataframe(chosen_game)
        st.write('These are recommended based on similarity among games')
        st.dataframe(results.iloc[:25])
        st.write(
            'These are recommended based on what other players have also '
            'played'
            )
        st.dataframe(results_2.iloc[:25])
