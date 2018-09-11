
from flask import request, Flask
import pandas as pd


app = Flask(__name__)


@app.route("/USD/", methods=['GET'])
def predictVolume():
    df = pd.read_csv("EURUSD_15m_BID_01.01.2010-31.12.2016.csv")
    df['Time'] = pd.to_datetime(df.Time)
    df.set_index('Time', inplace=True)
    df = df.resample('D').mean()
    df = df.shift(-1)
    date = request.args.get('date', default=None, type=str)
    if date != None:
        try:
            s = "<h1>That is the Prediction of the next day:</h1><table border='1px'>"
            s+="<tr>"
            for v in df:
                s+="<th>"+str(v)+"</th>"
            s+="</tr>"
            s+="<tr>"     
            s+="<td>"+str(df.loc[date]['Open'])+"</td>"
            s+="<td>"+str(df.loc[date]['Low'])+"</td>"
            s+="<td>"+str(df.loc[date]['High'])+"</td>"
            s+="<td>"+str(df.loc[date]['Close'])+"</td>"
            s+="<td>"+str(df.loc[date]['Volume'])+"</td>"
            s+="</tr>"
            s+="</table>"
        except:
            s = "<h1>This Date is unavilable, enter a date between 2010 and 2016</h1>"
        return str(s)
    else:
       
        s = "<h1>Please Enter a valid Date (YYYY-MM-DD)</h1>"
        return s


@app.route('/index/')
@app.route('/')
def index():
	return ('<form action="/USD/" method="get"><label>Date : </label><input type="date" name="q" /><input type="submit" value="Submit" /></form>')


if __name__ == "__main__":
	app.run(threaded=True)