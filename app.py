#!/usr/bin/env python

import pickle
import os
import sys
import re
from pathlib import Path
import logging
# from urllib import request

from flask import Flask, render_template, request
import basilica # type: ignore
from werkzeug import secure_filename
from numpy import array, arccos, pi, dot # type: ignore
from numpy.linalg import norm # type: ignore
from pandas import DataFrame # type: ignore
from scipy.spatial import distance # type: ignore
from sklearn.decomposition import PCA # type: ignore


SUBTITLE = os.getenv('SUBTITLE') # 'testing visual properties of hate symbols with magical algorithms'
STATIC='static'
BASILICA_KEY = os.getenv('BASILICA_KEY')# '503aaf17-3cc1-b7a3-d8b9-e9d080f207e5' # should be env var
PORT = int(os.environ.get('PORT', 5000))
#
def create_app():
    ''' create and configure an instance of the Flask application '''
    app = Flask(
        __name__,
        static_url_path=f'/{STATIC}',
        # instance_path=Path(f'{STATIC}').resolve()
    )
   
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///adlproj_db.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['ENV'] = 'debug' # TODO: Change beffore deploying
    app.config['UPLOAD_FOLDER'] = 'tmp'
    # create the folders when setting up your app
    # thanks https://stackoverflow.com/a/42425388/10993971
    os.makedirs(os.path.join(app.instance_path, app.config['UPLOAD_FOLDER']), exist_ok=True)

    @app.route('/')
    def home():

        return render_template('home.html', title=SUBTITLE)

    @app.route('/clustersimilarity', methods=['POST'])
    def clustersimilarity() -> str:

        f = request.files['img']
        uploaded_image = os.path.join(STATIC, app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(uploaded_image)

        with basilica.Connection(BASILICA_KEY) as c:
            embedding_ = c.embed_image_file(uploaded_image)

        with open(f'{STATIC}/model.pickle', 'rb') as model_pickle:
            model = pickle.load(model_pickle)

        with open(f'{STATIC}/dataframe.pickle', 'rb') as df_pickle:
            df = pickle.load(df_pickle)

        embedding = array([embedding_])
        prediction = model.predict(embedding)[0]

        similars = df[df.labels==prediction]

        similar_jpgs = [f'jpgs/hate-symbols-db-{idx:04d}.jpg' for idx in similars.index]

        return render_template(
            'clustersimilarity.html',
            title=SUBTITLE,
            prediction = prediction,
            filepath = uploaded_image,
            similar_jpgs = similar_jpgs
        )

    @app.route('/cosinesimilarity', methods=['POST'])
    def cosinesimilarity(THRESHOLD: float = .75) -> str:

        f = request.files['img']
        uploaded_image = os.path.join(STATIC,  app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(uploaded_image)

        with open(f'{STATIC}/dataframe.pickle', 'rb') as df_pickle:
            df = pickle.load(df_pickle)

        with basilica.Connection(BASILICA_KEY) as c:
            embedding = c.embed_image_file(uploaded_image, opts={'dimensions': df.shape[1]-1})

        training_embeddings = df.drop('labels', axis=1)

        def similarity(u: array, v: array) -> float:
            ''' https://en.wikipedia.org/wiki/Cosine_similarity'''
            numer = dot(u, v)
            denom = norm(u) * norm(v)

            return numer / denom


        similar_jpgs_ids = (
            idx
            for idx, tr_embedding
            in training_embeddings.T.iteritems()
            if  similarity(embedding, tr_embedding) > THRESHOLD
        )

        similar_jpgs =  [f'jpgs/hate-symbols-db-{idx:04d}.jpg' for idx in similar_jpgs_ids]

        return render_template(
            'cosinesimilarity.html',
            title = SUBTITLE,
            similar_jpgs = similar_jpgs,
            threshold = THRESHOLD,
            filepath = uploaded_image
        )
    
    return app

#if __name__=='__main__':
#    app.run(debug=True, host='0.0.0.0', port=PORT)
