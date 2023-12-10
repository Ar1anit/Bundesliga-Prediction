from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib 
from flask_cors import CORS
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)

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
    try: 
        print("Anfrage erhalten")
        game_data = request.json
        app.logger.info(f"Empfangene Daten: {game_data}")
        predictions = []
        rolling_averages = pd.read_csv("../datasets/rolling.csv")
        print(f"Empfangene Daten: {game_data}")
        rolling_averages.dropna(inplace=True)
        for data in game_data:
            team_code = team_name_to_code[data["HomeTeam"]]
            opp_code = team_name_to_code[data["AwayTeam"]]
            date_object = datetime.strptime(data['Date'], '%d-%m-%Y')
            day_code = date_object.weekday()
            hour = int(re.search(r'^(\d+):', data['Time']).group(1))

            model_input = [
                team_code, 
                opp_code, 
                hour, 
                day_code,
                data['B365H'], data['B365D'], data['B365A'],
                data['BWH'], data['BWD'], data['BWA'],
                data['IWH'], data['IWD'], data['IWA'],
                data['PSH'], data['PSD'], data['PSA'],
                data['WHH'], data['WHD'], data['WHA'],
                data['VCH'], data['VCD'], data['VCA'],
                data['MaxH'],data['MaxD'],data['MaxA'],

            ]

            relevant_averages = rolling_averages[(rolling_averages['team_codes'] == team_code) & (rolling_averages['opp_codes'] == opp_code)]

            if not relevant_averages.empty:
                model_input.extend([
                    relevant_averages["FTHG_rolling"].iloc[0], 
                    relevant_averages["FTAG_rolling"].iloc[0],
                    relevant_averages["HTHG_rolling"].iloc[0],
                    relevant_averages["HTAG_rolling"].iloc[0],
                    relevant_averages["HS_rolling"].iloc[0],
                    relevant_averages["AS_rolling"].iloc[0],
                    relevant_averages["HST_rolling"].iloc[0],
                    relevant_averages["AST_rolling"].iloc[0],
                    relevant_averages["HC_rolling"].iloc[0],
                    relevant_averages["AC_rolling"].iloc[0],
                    relevant_averages["HF_rolling"].iloc[0],
                    relevant_averages["AF_rolling"].iloc[0],
                    relevant_averages["HY_rolling"].iloc[0],
                    relevant_averages["AY_rolling"].iloc[0],
                    relevant_averages["HR_rolling"].iloc[0],
                    relevant_averages["AR_rolling"].iloc[0]
                ])
                required_keys = ['HomeTeam', 'AwayTeam', 'Date', 'Time', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'MaxH', 'MaxD', 'MaxA']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    predictions.append({'error': f'Fehlende Daten: {missing_keys}', 'HomeTeam': data.get('HomeTeam', 'Unbekannt'), 'AwayTeam': data.get('AwayTeam', 'Unbekannt')})
                    continue 

                app.logger.info(f"Erzeugter Modellinput für Spiel: {model_input}")

                probabilities = model.predict_proba([model_input])
                win_probability = probabilities[0][1]

                prediction = model.predict([model_input])[0]

                prediction_result = {
                'HomeTeam': data['HomeTeam'],
                'AwayTeam': data['AwayTeam'],
                'WinProbability': win_probability,
                'Prediction': 'Heimmannschaft gewinnt' if model.predict([model_input])[0] == 1 else 'Heimmannschaft gewinnt nicht'
                }

                prediction_result['Prediction'] = 'Heimmannschaft gewinnt' if prediction == 1 else 'Heimmannschaft gewinnt nicht'
                predictions.append(prediction_result)
            else:
                 predictions.append({
                    'error': 'Keine relevanten Durchschnittswerte gefunden.',
                    'HomeTeam': data['HomeTeam'],
                    'AwayTeam': data['AwayTeam']
                })
                
        if predictions:
            return jsonify(predictions), 200
        else:
            return jsonify({'error': 'Keine Spiele zur Vorhersage übermittelt.'}), 400
        
    except Exception as e:
        app.logger.error(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
