from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the Pokemon dataset
pokemon = pd.read_csv('pokemon_alopez247.csv')

# Prepare the data
X = pokemon[['Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed']]
y = pokemon.isLegendary

# Create and train the RandomForestClassifier model
model = RandomForestClassifier(max_leaf_nodes=10)
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        total = float(request.form.get('total'))
        hp = float(request.form.get('hp'))
        attack = float(request.form.get('attack'))
        defense = float(request.form.get('defense'))
        sp_atk = float(request.form.get('sp_atk'))
        sp_def = float(request.form.get('sp_def'))
        speed = float(request.form.get('speed'))

        # Make predictions using the trained model
        prediction = model.predict([[total, hp, attack, defense, sp_atk, sp_def, speed]])

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
