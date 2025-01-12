from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the dataset
df = pd.read_excel('chronic_diseases_recommendations_expanded copy.xlsx')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        age_range = request.form['age']  
        weight = request.form['weight']
        bp = request.form['bp']
        gender = request.form['gender'].strip().lower()
        disease = request.form['disease'].strip().lower()
        severity_of_pain = request.form['severity_of_pain'].strip().lower()
        
        print(f"Received Data: Name={name}, Age Range={age_range}, Weight={weight}, BP={bp}, Gender={gender}, Disease={disease}, Severity={severity_of_pain}")

        # Normalize the DataFrame columns for comparison
        df['Chronic Disease'] = df['Chronic Disease'].str.strip().str.lower()
        df['gender'] = df['gender'].str.strip().str.lower()
        df['severity_of_pain'] = df['severity_of_pain'].str.strip().str.lower()

        # Debugging: Print the unique values for the relevant columns
        print("Unique diseases in dataset: ", df['Chronic Disease'].unique())
        print("Unique genders in dataset: ", df['gender'].unique())
        print("Unique severities of pain in dataset: ", df['severity_of_pain'].unique())

        # Convert 'age' to str and strip any whitespaces in the DataFrame
        df['age'] = df['age'].astype(str).str.strip()

        
        recommendations = df[
            (df['Chronic Disease'] == disease) &
            (df['age'] == age_range) &  
            (df['gender'] == gender) &
            (df['severity_of_pain'] == severity_of_pain)
        ]

        
        print("Filtered Recommendations: ", recommendations)

        if not recommendations.empty:
            # Extract relevant recommendations
            diet_morning = recommendations['dietmorning'].dropna().tolist() or ['No morning diet recommendations available.']
            diet_afternoon = recommendations['dietafternoon'].dropna().tolist() or ['No afternoon diet recommendations available.']
            diet_night = recommendations['dietnight'].dropna().tolist() or ['No night diet recommendations available.']
            
            yoga_morning = recommendations['Yoga Recommendation (Pose 1)'].dropna().tolist() or ['No morning yoga recommendations available.']
            yoga_afternoon = recommendations['Yoga Recommendation (Pose 2)'].dropna().tolist() or ['No afternoon yoga recommendations available.']
            yoga_evening = recommendations['Yoga Recommendation (Pose 3)'].dropna().tolist() or ['No evening yoga recommendations available.']
            yoga_night = recommendations['Yoga Recommendation (Pose 4)'].dropna().tolist() or ['No night yoga recommendations available.']
        else:
            diet_morning = ['No morning diet recommendations available.']
            diet_afternoon = ['No afternoon diet recommendations available.']
            diet_night = ['No night diet recommendations available.']
            yoga_morning = ['No morning yoga recommendations available.']
            yoga_afternoon = ['No afternoon yoga recommendations available.']
            yoga_evening = ['No evening yoga recommendations available.']
            yoga_night = ['No night yoga recommendations available.']

        return render_template('results.html', name=name, disease=disease, 
                               diet_morning=diet_morning, diet_afternoon=diet_afternoon, diet_night=diet_night,
                               yoga_morning=yoga_morning, yoga_afternoon=yoga_afternoon, 
                               yoga_evening=yoga_evening, yoga_night=yoga_night)
    
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')

if __name__ == '__main__':
    app.run(debug=True)
