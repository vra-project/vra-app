
'''
Codigo para la creacion de la segunda aplicacion en Streamlit
'''

# %%
# Se cargan las librerias necesarias para realizar este proceso

import random as rd
import ast
import boto3
import pandas as pd
import numpy as np
import streamlit as st

# %%
# Se cargan las claves necesarias para utilizar a lo largo del proceso
# También se necesitan claves de acceso a nuestro servidor de AWS

BUCKET_S3 = st.secrets['bucket_s3']
NEW_FILE_NAME = st.secrets['new_file_name']
COMPLEX_NAME = st.secrets['complex_name']
CLEAN_FOLDER = st.secrets['clean_folder']
NOT_PLAYED = st.secrets['not_played']
NEW_REVIEWS = st.secrets['new_reviews']

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

PERC = 0.5
N_USERS = 10

MULTI_COLS = {
    'platforms': 'Select your platforms',
    'genres': 'Select your favorite genres',
    'themes': 'Select your favorite themes',
    'game_modes': 'Select how you want to play',
    'player_perspectives': 'Select your favorite perspectives'
    }

RATING_VALUE = {
    'Essential': 5,
    'Good Game': 4,
    'This Game Exists': 3,
    'Avoid': 1,
    "Haven't played": 0
    }

REVIEW_COLS = ['id', 'user_id', 'game_id', 'review_rating']

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

    if 'not_played_df' not in st.session_state:
        st.session_state.not_played_df = pd.read_feather(
            f'{BUCKET_S3}/clean_reviews/not_played.feather'
            )

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

games_df = st.session_state.games_df.copy()
complex_df = st.session_state.complex_df.copy()
id_review = max([max(st.session_state.reviews_df['id']), 8999999]) + 1
try:
    ID_NP = max(st.session_state.not_played_df['id']) + 1
except ValueError:
    ID_NP = 0

# %%
# Se definen ciertas funciones para la mejora del código


def send_new_reviews():
    '''
    Se cargan las nuevas reviews en el bucket S3
    '''
    (
     st.session_state.reviews_df
     .loc[st.session_state.reviews_df['id'] >= 9000000]
     .reset_index(drop=True)
     .to_feather(f'{BUCKET_S3}/{NEW_REVIEWS}')
     )


def send_not_played():
    '''
    Se cargan los juegos que no haya jugado un determinado usuario al bucket S3
    '''
    (
     st.session_state.not_played_df
     .reset_index(drop=True)
     .to_feather(f'{BUCKET_S3}/{NOT_PLAYED}')
     )


def concat_new_data(original_df, new_data):
    '''
    Se usara esta funcion para concatenar los nuevos datos a los originales
    '''
    new_df = pd.DataFrame(
        columns=REVIEW_COLS,
        data=[new_data]
        )
    return pd.concat([original_df, new_df])


# %%
# Se pide nombre de usuario
user_name = st.text_input('Enter your username').replace(' ', '-').upper()
st.write('The more reviews you do, the more accurate the software will be!')
if user_name != '':
    # Se cargan los juegos disponibles, para no darle opciones que ya haya
    # jugado
    user_reviews = st.session_state.reviews_df.loc[
        st.session_state.reviews_df['user_id'] == user_name
        ]
    not_played = st.session_state.not_played_df.loc[
        st.session_state.not_played_df['user_id'] == user_name
        ]
    user_games = (
        st.session_state.games_df
        .merge(pd.concat([user_reviews, not_played]), on='game_id')
        ['name']
        .tolist()
        )
    # Tambien se carga un dataframe para alojar aquellos juegos que el usuario
    # indique que no ha jugado, para que no se le pida una review de estos
    not_reviewed = (
        st.session_state.games_df
        .loc[~(
            st.session_state.games_df['name']
            .isin(user_games)
            )]
        [['name', 'RAWG_nreviews']]
        )
    not_reviewed = st.session_state.complex_df.merge(not_reviewed)
    # Se crea un formulario para pedir reviews de juegos pseudo aleatorios,
    # pues se seleccionara entre los 100 con mas reviews
    with st.form('random_review'):
        # Se muestra el titulo
        st.title("Review a random title!")
        example = (
            not_reviewed
            .sort_values('RAWG_nreviews', ascending=False)
            .drop('RAWG_nreviews', axis=1)
            .head(100).sample(1)
            .reset_index(drop=True)
            )
        example = example.rename(columns={
            col: col.title().replace('_', ' ')
            for col in example.columns
            })
        example = example.rename(columns={'Oc Rating': 'Rating'})
        example.index = [1]
        st.dataframe(example)
        # Se solicita una review
        score = RATING_VALUE[
            st.radio(
                'Review this game',
                RATING_VALUE.keys(),
                index=0
                )
            ]
        game_id = (
            example
            .merge(st.session_state.games_df, left_on='Name', right_on='name')
            [['game_id']]
            .iloc[0]
            ['game_id']
            )
        # Dado que streamlit reprocesa constantemente, se guardan los datos de
        # tal forma que la review se aloje de la forma correcta
        if 'results' not in st.session_state:
            st.session_state.results = [game_id]
        else:
            st.session_state.results.append(game_id)
        SEND_REVIEW = st.form_submit_button('Send Review')
        # Una vez enviada la review, se guardan los resultados
        if SEND_REVIEW:
            if score > 0:
                st.session_state.reviews_df = concat_new_data(
                    st.session_state.reviews_df,
                    [id_review, user_name, st.session_state.results[-2], score]
                    )
                send_new_reviews()
                user_reviews = (
                    st.session_state.reviews_df.loc[
                        st.session_state.reviews_df['user_id'] == user_name
                        ]
                    )
            else:
                st.session_state.not_played_df = concat_new_data(
                    st.session_state.not_played_df,
                    [ID_NP, user_name, st.session_state.results[-2], score]
                    )
                send_not_played()
            st.session_state.results = st.session_state.results[-1:]

    # Se crea un formulario para que el usuario de una reviews del titulo que
    # desee
    with st.form('selected review'):
        # Se solicita el titulo
        st.title("Review a title")
        game_name = st.selectbox(
            'Select a game',
            [''] + st.session_state.complex_df['name'].tolist(),
            key='game_2'
            )
        # Se solicita la review
        score = RATING_VALUE[
            st.radio(
                'Review this game',
                RATING_VALUE.keys(),
                index=0
                )
            ]
        if game_name != '':
            game_id = (
                st.session_state.games_df
                .loc[st.session_state.games_df['name'] == game_name, 'game_id']
                .iloc[0]
                )
        SEND_REVIEW = st.form_submit_button('Send Review')
        if SEND_REVIEW:
            # Se guardan los resultados, aunque saltara un error si no se ha
            # dado un nombre de juego
            if game_name != '':
                past_review = st.session_state.reviews_df.loc[
                    (st.session_state.reviews_df['user_id'] == user_name) &
                    (st.session_state.reviews_df['game_id'] == game_id)
                    ]
                null_review = st.session_state.not_played_df.loc[
                    (st.session_state.not_played_df['user_id'] == user_name) &
                    (st.session_state.not_played_df['game_id'] == game_id)
                    ]
                if len(past_review) == 1:
                    if score > 0:
                        st.session_state.reviews_df.loc[
                            (st.session_state.reviews_df['user_id'] ==
                             user_name) &
                            (st.session_state.reviews_df['game_id'] ==
                             game_id),
                            'review_rating'
                            ] = score
                    else:
                        st.session_state.reviews_df = (
                            st.session_state.reviews_df
                            .loc[
                                (st.session_state.reviews_df['user_id'] !=
                                 user_name) &
                                (st.session_state.reviews_df['game_id'] !=
                                 game_id)
                                ]
                            )
                        send_new_reviews()
                        st.session_state.not_played_df = concat_new_data(
                            st.session_state.not_played_df,
                            [ID_NP, user_name, game_id, score]
                            )
                        send_not_played()
                else:
                    if score > 0:
                        st.session_state.reviews_df = concat_new_data(
                            st.session_state.reviews_df,
                            [id_review, user_name, game_id, score]
                            )
                        send_new_reviews()
                        if len(null_review) > 0:
                            st.session_state.not_played_df = (
                                st.session_state.not_played_df
                                .loc[
                                    (st.session_state.not_played_df['user_id']
                                     != user_name) &
                                    (st.session_state.not_played_df['game_id']
                                     != game_id)
                                    ]
                                )
                            send_not_played()
                    elif len(null_review) == 0:
                        st.session_state.not_played_df = concat_new_data(
                            st.session_state.not_played_df,
                            [ID_NP, user_name, game_id, score]
                            )
                        send_not_played()
            else:
                st.error("You didn't select any games")
            user_reviews = (
                st.session_state.reviews_df.loc[
                    st.session_state.reviews_df['user_id'] == user_name
                    ]
                )

    if len(user_reviews) >= 10:
        # Si el usuario ha realizado mas de 10 reviews, se le invita a conocer
        # sus recomendaciones
        with st.form(key='user_filters'):
            st.title('Get your recommendations')
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
                            'Selecting none will mean that no filter is'
                            ' applied'
                            )
                        )
                    )

            # Se tiene en cuenta la retrcompatibilidad
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

            # Se pide una duracion
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
                        help=(
                            'Selecting 100 means you are choosing 100+ hours'
                            ' games'
                            )
                        )
                        )
            # Se pide un limite de edad
            with st.expander("Select an age limit if you'd like"):
                age = st.radio(
                    'Select an age limit',
                    (
                        st.session_state.complex_df['age_ratings']
                        .sort_values()
                        .unique()
                        ),
                    index=4
                    )

            # Se realizan las recomendaciones
            SEND_GAME = st.form_submit_button('Search recommendations')
            if SEND_GAME:
                with st.spinner(rd.choice(EASTER_EGGS)):
                    complex_df = st.session_state.complex_df.copy()
                    # Se filtran los resultados
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

                    complex_df = complex_df.loc[
                        complex_df['age_ratings'] <= age
                        ]

                    if len(complex_df) == 0:
                        st.warning(
                            "There aren't any games that match those filters"
                            )
                    else:
                        games_df = (
                            complex_df[['name']]
                            .merge(st.session_state.games_df, on='name')
                            )
                        similar_df = (
                            games_df
                            .drop([
                                'game_id', 'RAWG_rating', 'RAWG_nreviews'
                                ],
                                axis=1
                                )
                            )
                        # Se prepara el dataframe para las recomendaciones
                        pivoted_df = (
                            games_df
                            .loc[games_df['RAWG_nreviews'] > 4]
                            .merge(
                                st.session_state.reviews_df,
                                how='left',
                                on='game_id'
                                )
                            .pivot_table(
                                index='user_id',
                                columns='game_id',
                                values='review_rating'
                                )
                            .replace([1, 3, 4, 5], [-1, 0, 1, 5])
                            )


                        def user_based_recommender(user_slug):
                            '''
                            En base a un usuario, se buscaran usuarios
                            similares, teniendo en cuenta que compartan titulos
                            jugados y con las mismas valoraciones. En base a
                            esto, se le recomendaran juegos que les haya
                            gustado a otros usuarios
                            '''
                            # Se obtienen los juegos jugados por el usuario
                            user_df = pivoted_df.loc[
                                pivoted_df.index == user_slug
                                ]
                            user_played_games = (
                                user_df.columns[user_df.notna().any()].tolist()
                                )

                            pivoted_user_df = pivoted_df[user_played_games]
                            # Se obtienen los jugadores que hayan jugado a un
                            # porcentaje dado de los titulos jugados por el
                            # usuario
                            limited_user_df = pivoted_user_df.loc[
                                pivoted_user_df.count(axis=1) >=
                                (len(user_played_games) * PERC)
                                ]
                            # Se comparan los jugadores, y se obtienen los mas
                            # similares
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

                            top_related_users = corr_values.iloc[:N_USERS]
                            avg_corr = top_related_users['corr'].mean()
                            # Se crea un campo que indica la valoracion de cada
                            # usuario a un juego, por la similitud que tenga
                            # con el usuario original
                            top_related_rating = (
                                top_related_users.merge(
                                    st.session_state.reviews_df
                                    )
                                .assign(weighted_rating=lambda df:
                                        df['corr'] * df['review_rating'])
                                .drop('user_id', axis=1)
                                )
                            # Se recomiendan unicamente juegos que no haya
                            # jugado el usuario
                            non_played_rating = top_related_rating.loc[
                                ~(top_related_rating['game_id']
                                  .isin(user_played_games))
                                ]
                            # Aquellos juegos recomendados debe haberlos jugado
                            # cierto porcentaje de los usuarios top, y recibir
                            # una valoracion promedio positiva
                            recommendation_df = (
                                non_played_rating
                                .groupby('game_id')
                                ['weighted_rating']
                                .agg(['count', 'mean'])
                                .reset_index()
                                .loc[lambda df:
                                     (df['count'] >= N_USERS * PERC) &
                                     (df['mean'] >= 4*avg_corr)
                                     ]
                                .sort_values('mean', ascending=False)
                                .drop(['count'], axis=1)
                                .merge(
                                    games_df[['name', 'game_id']],
                                    on='game_id'
                                    )
                                [['name', 'mean']]
                                )

                            return recommendation_df


                    # Se da un formato adecuado a los juegos recomendados
                    complex_df = complex_df.rename(columns={
                        col: col.title().replace('_', ' ')
                        for col in complex_df.columns
                        })
                    complex_df = complex_df.rename(
                        columns={'Oc Rating': 'Rating'}
                        )
                    results = (
                        user_based_recommender(user_name)
                        [['name']]
                        .merge(complex_df, left_on='name', right_on='Name')
                        .drop('name', axis=1)
                        .iloc[:10]
                        )
                    if len(results) > 0:
                        results.index = range(1, len(results)+1)
                        st.write(
                            'These are recommended based on similarity among '
                            'users'
                            )
                        st.dataframe(results)
                    else:
                        st.write(
                            "There wasn't any match, try reviewing more games"
                            )

    else:
        st.text(
            'You need at least 10 reviews to get recommendations and you did '
            f'{len(user_reviews)}'
            )
        