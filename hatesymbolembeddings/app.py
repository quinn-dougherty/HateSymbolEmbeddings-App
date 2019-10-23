#!/usr/bin/env python

import pickle
import os
import sys
import re
from pathlib import Path
import logging
from typing import Tuple
# from urllib import request

from flask import Flask, render_template, request
import basilica # type: ignore
from werkzeug.utils import secure_filename
from numpy import array, arccos, pi, dot # type: ignore
from numpy.linalg import norm # type: ignore
from pandas import DataFrame # type: ignore
from scipy.spatial import distance # type: ignore
from sklearn.decomposition import PCA # type: ignore

from .settings import TITLE, SUBTITLE, BASILICA_KEY, PORT

# SUBTITLE = os.getenv('SUBTITLE') # 'testing visual properties of hate symbols with magical algorithms'
STATIC='static'
#BASILICA_KEY = os.getenv('BASILICA_KEY')
# PORT = int(os.environ.get('PORT', 33507))
#
TMP = 'tmp'

def get_upload_save_it_and_return_embedding() -> Tuple[array, str]: 
    f = request.files['img']
    uploaded_image = os.path.join(STATIC, TMP, secure_filename(f.filename))
    f.save(uploaded_image)

    with basilica.Connection(BASILICA_KEY) as c:
        embedding_ = c.embed_image_file(uploaded_image)

    return embedding_, uploaded_image


def create_app():
    ''' create and configure an instance of the Flask application '''
    app = Flask(
        __name__,
        static_url_path=f'/{STATIC}',
        # instance_path=Path(f'{STATIC}').resolve()
    )
   
    # app.config['ENV'] = 'debug' # TODO: Change beffore deploying
    app.config['UPLOAD_FOLDER'] = TMP
    os.makedirs(os.path.join(STATIC, app.config['UPLOAD_FOLDER']), exist_ok=True)

    @app.route('/')
    def home():

        return render_template('home.html', title=TITLE, subtitle=SUBTITLE)

    @app.route('/clustersimilarity', methods=['POST'])
    def clustersimilarity() -> str:
        
        embedding_, uploaded_image = get_upload_save_it_and_return_embedding()

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
            title=TITLE,
            subtitle=SUBTITLE,
            prediction = prediction,
            filepath = uploaded_image,
            similar_jpgs = similar_jpgs
        )

    @app.route('/cosinesimilarity', methods=['POST'])
    def cosinesimilarity(THRESHOLD: float = .75) -> str:

        def similarity(u: array, v: array) -> float:
            ''' https://en.wikipedia.org/wiki/Cosine_similarity'''
            numer = dot(u, v)
            denom = norm(u) * norm(v)

            return numer / denom

        with open(f'{STATIC}/dataframe.pickle', 'rb') as df_pickle:
            df = pickle.load(df_pickle)

        embedding, uploaded_image = get_upload_save_it_and_return_embedding()

        training_embeddings = df.drop('labels', axis=1)
        similar_jpgs_ids = (
            idx
            for idx, tr_embedding
            in training_embeddings.T.iteritems()
            if  similarity(embedding, tr_embedding) > THRESHOLD
        )

        similar_jpgs =  [f'jpgs/hate-symbols-db-{idx:04d}.jpg' for idx in similar_jpgs_ids]

        return render_template(
            'cosinesimilarity.html',
            subtitle=SUBTITLE,
            title = TITLE,
            similar_jpgs = similar_jpgs,
            threshold = THRESHOLD,
            filepath = uploaded_image
        )
    
    @app.route('/knn', methods=['POST'])
    def knn() -> str: 
        pass

    @app.route('/howwhy')
    def howwhy() -> str: 
        return render_template('howwhy.html', subtitle=SUBTITLE, title=TITLE)

    return app

import logging
import sys
from logging import Formatter

def log_to_stderr(app):
  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
  ))
  handler.setLevel(logging.WARNING)
  app.logger.addHandler(handler)

#if __name__=='__main__':
#    app = create_app()
#    log_to_stderr(app)
#    app.run(debug=True, host='0.0.0.0', port=PORT)
