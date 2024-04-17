from flask import Flask, request, render_template
import joblib
import sqlite3

app = Flask(__name__)

# Load the trained model
model = joblib.load('Model/decision_tree.pkl')

# Connect to the SQLite database
conn = sqlite3.connect('reviews.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS reviews
                  (review TEXT, prediction TEXT)''')
conn.commit()

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for classification
@app.route('/prediction', methods=['get', 'post'])
def prediction():
    # Get the review text from the form
    rw = request.form.get('review')

    # Perform classification
    prediction = model.predict([rw])[0]

    # Map prediction to 'good' or 'bad'
    result = 'good' if prediction == 1 else 'bad'

    # Store the review and prediction in the database
    cursor.execute("INSERT INTO reviews (review, prediction) VALUES (?, ?)", (rw, result))
    conn.commit()

    # Return the classification result
    return render_template('output.html', result=result)

######################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
