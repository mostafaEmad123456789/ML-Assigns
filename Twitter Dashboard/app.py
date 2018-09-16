from flask import request, Flask, render_template
import os
from sklearn.externals import joblib
from twitterscraper import query_tweets
import matplotlib.pyplot as plt
import io
import base64
import re
import io
from threading import Thread

app = Flask(__name__)


@app.route("/tweetSearch/", methods=['GET'])
def tweetSearch():
    q = request.args.get('q', default=None, type=str)
    s=""
    p_count = 0
    n_count = 0
    if q:
        pattern = r'http://[a-z\W+.A-Z\W+]+|#\w+|[a-zA-Z]+|[.]+|[\n0-9\xa0]+|[_\W]+'
        regex = re.compile(pattern)
        tweets = []
        for tweet in query_tweets(q, 10, lang="ar")[:10]:
            tweet.text =regex.sub(" ", tweet.text.strip())
            tweets.append(tweet)
        for tweet in tweets:
            text = tweet.text
            try:
                RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
                PICKLE_DIR = os.path.join(RES_DIR, 'model.pkl')
                VECTORIZER_DIR = os.path.join(RES_DIR, 'vectorizer.pkl')
                clf = joblib.load(PICKLE_DIR)
                vectorizer = joblib.load(VECTORIZER_DIR)
                q_vect = vectorizer.transform([text])
                y_pred = clf.predict(q_vect)
                if y_pred[0] == "Positive":
                    p_count+=1
                elif y_pred[0] == "Negative":
                    n_count+=1
                
            except Exception as e:
                s = "<h1>"+str(e)+"</h1>"

        sizes = [p_count, n_count]
        labels = ["Positive", "Negative"]
        plt.pie(sizes, labels=labels, autopct='%1.2f%%', shadow=True)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot = base64.b64encode(img.getvalue()).decode()
        s = "<h1>Positive : " + str(p_count) + ", Negative : " + str(n_count) + "</h1>"
        #return s
        return render_template('dashboard.html', graph = plot)
    else:
        return "<h1>Enter a Search Phrase.</h1>"
    

@app.route('/starttask', methods=['GET'])
def start_task():
    def do_work(value):
        x = value
        x+=1
        import time
        time.sleep(20)
        return render_template('dashboard.html')
    
    value = request.args.get('value', default=None, type=int)
    thread = Thread(target=do_work, kwargs={'value': value})
    thread.start()
    return render_template('dashboard.html', graph=value)


@app.route('/index')
@app.route('/')
def index():
	return ('<form action="/tweetSearch/" method="get"><label>Search Phrase : </label><input type="text" name="q" /><input type="submit" value="Submit" /></form>')

if __name__ == "__main__":
	app.run(debug=True)