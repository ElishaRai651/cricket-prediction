from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models
victory_pipe = pickle.load(open('pipe_win.pkl', 'rb'))
score_pipe = pickle.load(open('pipe_score.pkl', 'rb'))

teams = [
    'Mumbai Indians',
    'Chennai Super Kings',
    'Royal Challengers Bangalore',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Sunrisers Hyderabad',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

cities = [
    'Ahmedabad', 'Chennai', 'Mumbai', 'Bengaluru', 'Delhi', 'Dharamsala', 
    'Hyderabad', 'Lucknow', 'Jaipur', 'Chandigarh', 'Guwahati', 'Kolkata', 
    'Navi Mumbai', 'Pune', 'Dubai', 'Abu Dhabi', 'Sharjah', 'Visakhapatnam', 
    'Indore', 'Kanpur', 'Bangalore', 'Rajkot', 'Raipur', 'Ranchi', 
    'Cuttack', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban', 
    'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London', 'Cape Town'
]

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/victory_predictor', methods=['GET', 'POST'])
def victory_predictor():
    batting_team_prob = None
    bowling_team_prob = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data['batting_team'] = request.form['batting_team']
            form_data['bowling_team'] = request.form['bowling_team']
            form_data['city'] = request.form['city']
            form_data['total_score'] = int(request.form['total_score'])
            form_data['current_score'] = int(request.form['current_score'])
            form_data['overs'] = float(request.form['overs'])
            form_data['wickets'] = int(request.form['wickets'])

            # Calculations
            runs_left = form_data['total_score'] - form_data['current_score']
            balls_left = 120 - int(form_data['overs'] * 6)
            wickets_left = 10 - form_data['wickets']
            crr = form_data['current_score'] / form_data['overs'] if form_data['overs'] > 0 else 0
            rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [form_data['batting_team']],
                'bowling_team': [form_data['bowling_team']],
                'city': [form_data['city']],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'total_score': [form_data['total_score']],
                'crr': [crr],
                'rrr': [rrr]
            })

            result = victory_pipe.predict_proba(input_df)
            batting_team_prob = round(result[0][1] * 100, 2)
            bowling_team_prob = round(result[0][0] * 100, 2)

        except Exception as e:
            batting_team_prob = bowling_team_prob = f"Error: {e}"

    return render_template('index2.html', teams=sorted(teams), cities=sorted(cities), batting_team_prob=batting_team_prob, bowling_team_prob=bowling_team_prob, form_data=form_data)

@app.route('/score_predictor', methods=['GET', 'POST'])
def score_predictor():
    prediction = None
    form_data = {}

    if request.method == 'POST':
        try:
            form_data['batting_team'] = request.form['batting_team']
            form_data['bowling_team'] = request.form['bowling_team']
            form_data['city'] = request.form['city']
            form_data['current_score'] = int(request.form['current_score'])
            form_data['overs'] = float(request.form['overs'])
            form_data['wickets'] = int(request.form['wickets'])
            form_data['last_five_over_score'] = int(request.form['last_five_over_score'])
            form_data['last_five_over_wicket'] = int(request.form['last_five_over_wicket'])

            # Calculations
            balls_left = 120 - int(form_data['overs'] * 6)
            wickets_left = 10 - form_data['wickets']
            crr = form_data['current_score'] / form_data['overs'] if form_data['overs'] > 0 else 0

            input_df = pd.DataFrame({
                'batting_team': [form_data['batting_team']],
                'bowling_team': [form_data['bowling_team']],
                'city': [form_data['city']],
                'current_score': [form_data['current_score']],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'crr': [crr],
                'last_five_over_score': [form_data['last_five_over_score']],
                'last_five_over_wicket': [form_data['last_five_over_wicket']]
            })

            result = score_pipe.predict(input_df)
            prediction = int(result[0])

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index3.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
