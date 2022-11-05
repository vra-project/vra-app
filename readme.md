# VRA APP - Videogame Recommender Algorithm
Streamlit app designed to allow the users to interact with the algorithm and get different games recommendations. It's based on Python 3.9 and works with AWS.

## Table of contents
* [General info] (#general-info)
* [Technologies] (#technologies)S
* [Setup] (#setup)

## General info
This project downloads the games files and reviews files stored in S3 bucket.
Done this, the ui gets available for the user so they can interact with the 2 available pages.
Game Based page recommends games to the user based in a game title they introduce. The recommendation is based, first, in game similarity, and second, in games that others users have also played.
User Based page recommends games based in their different reviews, finding similar users to them and recommending them games these users like. All reviews done by the user are storaged in a S3 bucket.

## Technologies
Project is created with:
* Python 3.9
* Pandas 1.4.4
* Scipy 1.9.3
* S3fs 2022.10.0
* Streamlit 1.14.0

## Setup
To run this project, you'll need to install the libraries noted in requirements.txt.
This project is made to work inside AWS.
A file named secrets.toml containing the S3 Bucket name isn't uploaded.