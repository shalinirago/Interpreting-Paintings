import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from style import StyleClassification
from sentences import SentenceGeneration

# Define a flask app

app = Flask(__name__)


def gpt2_generate(predictions):
    
    #Function to generate sentences using the trained GPT2 model
    
    generator = SentenceGeneration(r'\output')
    sentences = ""
    sentences += generator.generate_sentence("The Painting is of the style " + str(predictions[1][0]))
    sentences += generator.generate_sentence("The Painting reminds me of")
    return sentences



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        classifier = StyleClassification(r'models\context_model_final_all.h5')
        predictions = classifier.predict(file_path)
        sentences = gpt2_generate(predictions)
        return sentences
        #return str(predictions[1][0])
    return None


if __name__ == '__main__':
    app.run(debug=True)

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=8080, debug=True)


