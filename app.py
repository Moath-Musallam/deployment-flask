import numpy as np
from flask import Flask, request,render_template
import pickle

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
   
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output =prediction[0]
    if output ==0:
        output ='Iris-setosa'
    elif output ==1:
        output ='Iris-versicolor'
    elif output ==2:
        output ='Iris-virginica'

    return render_template('index.html', prediction_text='The Flower is {}'.format(output))


if __name__ == '__main__':      # if name = main the app will start 
    app.run(debug=True)