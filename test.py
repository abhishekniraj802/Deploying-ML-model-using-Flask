from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open(r"C:/Users/RasAlGhul/GreatLakes/Hackathon/Kaggle/newfolder/caesarean.pkl", 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    dict = {'data1':[data1], 'data2': [data2], 'data3':[data3], 'data4':[data4], 'data5':[data5]}
    arr = pd.DataFrame(dict)
    pred = model.predict(arr)
    return render_template("after.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)

