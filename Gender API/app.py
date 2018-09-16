from flask import request, Flask
import os
from sklearn.externals import joblib


app = Flask(__name__)


@app.route("/gender/", methods=['GET'])
def gender():
    q = request.args.get('q', default=None, type=str)
    s=""
    if q:
        try:
            RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
            PICKLE_DIR = os.path.join(RES_DIR, 'model.pkl')
            ENCODER_DIR = os.path.join(RES_DIR, 'encoder.pkl')
            clf = joblib.load(PICKLE_DIR)
            encoder = joblib.load(ENCODER_DIR)
            first_alpha = q[0]
            last_alpha = q[0]
            length = len(q)
            q_first_alpha = encoder.transform([first_alpha])
            q_last_alpha = encoder.transform([last_alpha])
            y_pred = clf.predict([[q_first_alpha, q_last_alpha, length]])
            if y_pred[0] == "M":
                s = "<h1>Male</h1>"
            elif y_pred[0] == "F":
                s = "<h1>Female</h1>"
            
        except Exception as e:
            s = "<h1>"+str(e)+"</h1>"
        return s
    else:
        return "<h1>Enter a Name.</h1>"
    


@app.route('/index')
@app.route('/')
def index():
	return ('<form action="/gender/" method="get"><label>Name : </label><input type="text" name="q" /><input type="submit" value="Submit" /></form>')

if __name__ == "__main__":
	app.run(threaded=True)