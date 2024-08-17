from flask import Flask, render_template, request, jsonify,send_from_directory
import csv
from utils import audit_strategique_list,auditdigitale,create_radar_chart

app = Flask(__name__)

# Function to read questions from the CSV file by ID
def get_question_by_id(question_id):
    try:
        with open('questions.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['ID']) == question_id:
                    return row['question']
    except Exception as e:
        print(f"Error reading questions file: {e}")
    return None

@app.route('/')
def home():
    return render_template('audit_strategique.html')
@app.route('/audit_strategique', methods=['GET'])
def audit_strat_get():
    return render_template('audit_strategique.html')
@app.route('/audit_strategique', methods=['POST'])
def audit_strategique():
    try:
        data = request.get_json()
        question_ids = data.get('questions', [])
        answers = data.get('answers', [])
        
        if len(question_ids) != len(answers):
            return jsonify({"message": "Mismatch between number of questions and answers", "status": "error"}), 400
        
        # Process the questions and answers
        questions = [i  for i in range(1,18)]
        
        if None in questions:
            return jsonify({"message": "One or more questions not found", "status": "error"}), 400
        
        scores = audit_strategique_list(answers, questions)
        
        # Return a success response
        return jsonify({"message": "Answers processed successfully", "status": "success", "data": scores}), 200
    
    except Exception as e:
        # Handle any errors that occur
        print(f"Error in audit_strategique: {e}")
        return jsonify({"message": "An error occurred", "error": str(e), "status": "error"}), 500
@app.route('/audit_digitale', methods=['GET'])
def audit_digitale_get():
    return render_template('audit_digitale.html')
@app.route('/audit_digitale', methods=['POST'])
def audit_digitale_post():
    try:
        data = request.get_json()
        #question_ids = data.get('questions', [])
        answers = data.get('answers', [])
        print(answers)
        #if len(question_ids) != len(answers):
        #    return jsonify({"message": "Mismatch between number of questions and answers", "status": "error"}), 400
        
        # Process the questions and answers
        questions = [i  for i in range(1,17)]
        
        if None in questions:
            return jsonify({"message": "One or more questions not found", "status": "error"}), 400
        
        scores = auditdigitale(answers, questions)
        print(scores)
        create_radar_chart(scores)
        # Return a success response
        return jsonify({"message": "Answers processed successfully", "status": "success", "data": scores}), 200
    
    except Exception as e:
        # Handle any errors that occur
        print(f"Error in audit_digitale: {e}")
        return jsonify({"message": "An error occurred", "error": str(e), "status": "error"}), 500

@app.route('/get_image/<filename>')
def get_image(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)


@app.errorhandler(404)
def not_found(error):
    return '404 Not Found', 404

if __name__ == '__main__':
    app.run(debug=True)
