from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Model/decision_tree.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define a route for classification
@app.route('/classify', methods=['POST'])
def classify_review():
    # Get the review text from the form
    review_text = request.form.get('review')

    # Perform classification
    prediction = model.predict([review_text])[0]

    # Map prediction to 'good' or 'bad'
    result = 'good' if prediction == 1 else 'bad'
    
    # Return the classification result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)