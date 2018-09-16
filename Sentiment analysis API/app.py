from flask import request, Flask, jsonify
import os
from sklearn.externals import joblib


app = Flask(__name__)


@app.route("/PositiveOrNegative/", methods=['GET'])
def PositiveOrNegative():
    q = request.args.get('q', default=None, type=str)
    s=""
    score = {"Positive": 0,"Negative": 0}
    if q:
        try:
            RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
            PICKLE_DIR = os.path.join(RES_DIR, 'model.pkl')
            VECTORIZER_DIR = os.path.join(RES_DIR, 'vectorizer.pkl')
            clf = joblib.load(PICKLE_DIR)
            vectorizer = joblib.load(VECTORIZER_DIR)
            q_vect = vectorizer.transform([q])
            y_pred = clf.predict_proba(q_vect)
            #if y_pred[0] == "Positive":
             #   s = "<h1>Positive</h1>"
            #elif y_pred[0] == "Negative":
             #   s = "<h1>Negative</h1>"
            score = {"Positive": y_pred[0][1],"Negative": y_pred[0][0]}
            return jsonify(score)
        except Exception as e:
            s = "<h1>"+str(e)+"</h1>"
        return s
    else:
        return "<h1>Enter a Phrase.</h1>"
    


@app.route('/index')
@app.route('/')
def index():
	return ('<form action="/PositiveOrNegative" method="get"><label>Phrase : </label><input type="text" name="q" /><input type="submit" value="Submit" /></form>')

if __name__ == "__main__":
	app.run(threaded=True)