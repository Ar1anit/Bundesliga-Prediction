from flask import Flask, request, jsonify
from flask import render_template
app = Flask(__name__)
from ml_utils import prepare_features, matches, predictors, gbm_filtered

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    home_team = content['home_team']
    away_team = content['away_team']
    
    features_for_prediction = prepare_features(home_team, away_team, predictors, matches)
    
    predicted_proba = gbm_filtered.predict_proba(features_for_prediction)
    
    home_win_proba = predicted_proba[0][0]
    draw_proba = predicted_proba[0][1]
    away_win_proba = predicted_proba[0][2]
    
    return jsonify({'home_win_proba': home_win_proba, 'draw_proba': draw_proba, 'away_win_proba': away_win_proba})


if __name__ == '__main__':
    app.run(debug=True)
