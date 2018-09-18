from flask import request, Flask, jsonify
import numpy as np
from keras.models import load_model
from skimage import io
from scipy import misc
import os


app = Flask(__name__)



@app.route("/upload", methods=['POST'])
def upload():
		APP_ROOT = os.path.dirname(os.path.abspath(__file__))
		target = os.path.join(APP_ROOT, 'images')
		destination = ""
		print("target : ",target)

		if not os.path.isdir(target):
			os.mkdir(target)

		print("length : ", len(request.files.getlist("file")))
		
		for file in request.files.getlist("file"):
			print("file : ",file)
			filename = file.filename
			destination = "/".join([target, filename])
			print("dest : ",destination)
			file.save(destination)
			
		s=""
		try:
			model = load_model('model/my_model.h5')
			im = io.imread(destination, as_grey=True)
			im = misc.imresize(im , (240, 240))
			im = np.array(im)
			im = im.astype('float32')
			im /= 255
			im = np.array(im).reshape(-1, 240,240,1)
			Results = model.predict(im)
			print(Results)
			s = {"EOSINOPHIL":str(Results[0][0]), "LYMPHOCYTE":str(Results[0][1]), "MONOCYTE":str(Results[0][2]), "NEUTROPHIL":str(Results[0][3])}

		except Exception as e:
			s = {"Error": str(e)}
		return jsonify(s)
    


@app.route('/index')
@app.route('/')
def index():
	return ('<form action="/upload" method="post" enctype= "multipart/form-data"><label>Image : </label><input type="file" name="file" /><input type="submit" value="Submit" /></form>')

if __name__ == "__main__":
	app.run(threaded=True)