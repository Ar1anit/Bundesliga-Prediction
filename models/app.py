from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib 

app = Flask(__name__)

model = joblib.load('model.pkl')

team_name_to_code = {
    'Augsburg': 0, 
    'Bayern Munich': 1, 
    'Bielefeld': 2, 
    'Bochum': 3, 
    'Darmstadt': 4, 
    'Dortmund': 5,
    'Ein Frankfurt': 6, 
    'FC Koln': 7, 
    'Fortuna Dusseldorf': 8, 
    'Freiburg': 9,
    'Greuther Furth': 10, 
    'Heidenheim': 11, 
    'Hertha': 12, 
    'Hoffenheim': 13, 
    'Leverkusen': 14,
    "M'gladbach": 15, 
    'Mainz': 16, 
    'Paderborn': 17, 
    'RB Leipzig': 18, 
    'Schalke 04': 19, 
    'Stuttgart': 20,
    'Union Berlin': 21, 
    'Werder Bremen': 22, 
    'Wolfsburg': 23
}

day_code_to_weekday = {
    0: 'Montag',
    1: 'Dienstag',
    2: 'Mittwoch',
    3: 'Donnerstag',
    4: 'Freitag',
    5: 'Samstag',
    6: 'Sonntag'
}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data['HomeTeam'] = team_name_to_code[data['HomeTeam']]
    data['AwayTeam'] = team_name_to_code[data['AwayTeam']]
    data['Date'] = data['Date']["Date"].dt.day_of_week
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
