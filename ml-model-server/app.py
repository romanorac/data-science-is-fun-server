import logging

import eli5
from flask import Flask, render_template, request
from joblib import load

import logging_conf

logger = logging.getLogger()

CLF_PIPE = load('model.joblib')
TFIDF = CLF_PIPE.steps[0][1]
CLF = CLF_PIPE.steps[1][1]


def create_app():
    app = Flask(__name__)
    logging_conf.init_logging()
    return app


app = create_app()


@app.route('/')
def index():
    return render_template('index.html', name='Index')


@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    explain = eli5.show_prediction(CLF, doc=title, vec=TFIDF, target_names=['Bad', 'Good'],
                                   targets=['Good'])
    return render_template('response.html', name='Response', title=title, explain=explain)


if __name__ == '__main__':
    app.run(debug=False)
