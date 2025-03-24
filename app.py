from flask import Flask, render_template, request
import pickle
import numpy as np
import csv

app = Flask(__name__)


with open('linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)


dataset = []
with open('Air_Quality1.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)  
    for row in csvreader:
        dataset.append(row)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind-speed'])
            pollutant_emissions = float(request.form['pollutant-emissions'])

            
            input_data = np.array([[temperature, humidity, wind_speed, pollutant_emissions]])
            prediction = model.predict(input_data)[0]

            
            air_quality_index = get_air_quality_index(temperature, humidity, wind_speed, pollutant_emissions)

            
            return render_template('index.html', prediction=prediction, air_quality_index=air_quality_index)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        
        return render_template('index.html')


def get_air_quality_index(temperature, humidity, wind_speed, pollutant_emissions):
    for row in dataset:
        if (float(row[0]) == temperature and 
            float(row[1]) == humidity and 
            float(row[2]) == wind_speed and 
            float(row[3]) == pollutant_emissions):
            return row[4]  
    return "Not Found"  
if __name__ == '__main__':
   app.run(debug=True)