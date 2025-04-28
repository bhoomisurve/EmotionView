from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading
import time
import os
import google.generativeai as genai
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB Atlas connection
mongo_uri = os.getenv('MONGO_URI', ' ')
client = MongoClient(mongo_uri)
db = client['interview_app']
interviews_collection = db['interviews']
candidates_collection = db['candidates']
emotions_collection = db['emotion_data']

# Initialize GenAI client
genai_api_key = os.getenv('GENAI_API_KEY', '')
genai.configure(api_key=genai_api_key)

# Load the generative model
generation_model = genai.GenerativeModel('gemini-pro')

# Classes for 7 emotional states
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the pre-trained emotion model
model_best = load_model('face_model.h5')

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
camera = None
current_emotion = "Neutral"
interview_in_progress = False
emotion_history = []
current_question = "Press Start Interview to begin"
question_number = 0
interview_summary = ""
current_interview_id = None
question_answers = []
current_candidate = {
    "name": "",
    "email": "",
    "position": "general"
}

# Interview questions based on roles
interview_questions = {
    "general": [
        "Tell me about yourself.",
        "What are your greatest strengths?",
        "What do you consider to be your weaknesses?",
        "Why do you want this job?",
        "Where do you see yourself in five years?",
        "Why should we hire you?",
        "What is your greatest professional achievement?",
        "How do you handle stress and pressure?",
        "Describe a difficult work situation and how you overcame it.",
        "What are your salary expectations?"
    ],
    "software_engineer": [
        "Explain a complex technical concept in simple terms.",
        "Describe a challenging project you worked on.",
        "How do you stay updated with the latest technologies?",
        "How do you approach debugging a complex issue?",
        "Explain your experience with version control systems.",
        "How do you handle code reviews?",
        "How do you approach testing your code?",
        "Describe your experience with agile methodologies.",
        "How would you design a system that handles millions of users?",
        "What programming languages are you proficient in and why do you prefer them?"
    ],
    "data_scientist": [
        "Explain the difference between supervised and unsupervised learning.",
        "How would you handle imbalanced data?",
        "Describe a data project you worked on from start to finish.",
        "How do you validate your models?",
        "Explain overfitting and how to prevent it.",
        "What tools do you use for data visualization?",
        "How do you communicate technical findings to non-technical stakeholders?",
        "Describe your experience with big data technologies.",
        "How do you approach feature selection?",
        "What statistical methods do you typically use in your analysis?"
    ],
    "marketing": [
        "Describe a successful marketing campaign you developed.",
        "How do you measure the success of marketing initiatives?",
        "What tools do you use for marketing analytics?",
        "How do you stay updated with the latest marketing trends?",
        "Describe your experience with content marketing.",
        "How do you approach target audience analysis?",
        "What experience do you have with digital marketing platforms?",
        "Describe your approach to brand development.",
        "How do you handle social media marketing?",
        "What strategies would you use to increase our market share?"
    ]
}

def generate_ai_question(role, emotion=None, question_history=None):
    """Generate a question using GenAI based on detected emotion and previous questions"""
    if not question_history:
        question_history = []
    
    prompt = f"""
    You are an AI interviewer for a {role} position. 
    The candidate's current emotional state is: {emotion if emotion else 'Unknown'}.
    
    Previous questions asked: {', '.join(question_history) if question_history else 'None'}
    
    Generate a thoughtful, challenging interview question appropriate for this role and considering their current emotional state. 
    If they appear stressed or negative, ask a more supportive question. If they appear confident or positive, you may ask a more challenging question.
    
    Return ONLY the question text with no additional commentary.
    """
    
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating AI question: {e}")
        # Fallback to predefined questions
        return interview_questions[role][min(len(question_history), len(interview_questions[role])-1)]

def generate_interview_summary(role, emotion_log, question_answers=None):
    """Generate a summary of the interview using GenAI"""
    if not question_answers:
        question_answers = []
    
    # Format the inputs properly for the AI model
    emotion_summary = "\n".join([f"- At {timestamp}: {emotion}" for timestamp, emotion in emotion_log])
    
    qa_text = ""
    for i, qa in enumerate(question_answers):
        question = qa.get('question', 'Unknown question')
        answer = qa.get('answer', 'No answer provided')
        qa_text += f"\nQ{i+1}: {question}\nA: {answer}\n"
    
    prompt = f"""
    Generate a comprehensive interview summary for a {role} position candidate.
    
    Emotional responses during interview:
    {emotion_summary}
    
    Questions and answers:
    {qa_text}
    
    Based on the emotional responses and answers, provide insights about:
    1. The candidate's stress management
    2. Their emotional responses to different types of questions
    3. Overall emotional stability during the interview
    4. Communication skills and answer quality
    5. Recommendations for the hiring team
    
    Format the response as a professional interview summary.
    """
    
    try:
        response = generation_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating interview summary: {e}")
        return "Unable to generate interview summary. Please check the logs for more information."
    
    
def gen_frames():
    global camera, current_emotion, emotion_history, current_interview_id
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y + h, x:x + w]
            
            # Resize the face image to the required input size for the model
            try:
                face_image = cv2.resize(face_roi, (48, 48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = image.img_to_array(face_image)
                face_image = np.expand_dims(face_image, axis=0)
                face_image = np.vstack([face_image])
                
                # Predict emotion using the loaded model
                predictions = model_best.predict(face_image)
                emotion_label = class_names[np.argmax(predictions)]
                current_emotion = emotion_label
                
                if interview_in_progress:
                    timestamp = time.strftime("%H:%M:%S")
                    emotion_history.append((timestamp, emotion_label))
                    
                    # Save emotion data to MongoDB in real-time
                    if current_interview_id:
                        emotion_data = {
                            "interview_id": current_interview_id,
                            "timestamp": datetime.now(),
                            "emotion": emotion_label,
                            "confidence": float(np.max(predictions)),
                            "all_emotions": {class_names[i]: float(prob) for i, prob in enumerate(predictions[0])}
                        }
                        emotions_collection.insert_one(emotion_data)
                
                # Get the probabilities
                probs = predictions[0]
                
                # Display emotion probabilities as bar charts
                chart_height = 100
                chart_width = 150
                chart_x = x + w + 10
                chart_y = y
                
                # Draw background for the chart
                cv2.rectangle(frame, (chart_x, chart_y), 
                             (chart_x + chart_width, chart_y + chart_height), 
                             (240, 240, 240), -1)
                
                # Draw the bars
                bar_width = chart_width // len(class_names)
                for i, prob in enumerate(probs):
                    bar_height = int(prob * chart_height)
                    
                    # Color based on emotion
                    color = (120, 120, 120)  # Default gray
                    if class_names[i] == 'Happy':
                        color = (0, 255, 0)  # Green
                    elif class_names[i] in ['Angry', 'Disgusted', 'Fear']:
                        color = (0, 0, 255)  # Red
                    elif class_names[i] == 'Surprise':
                        color = (255, 165, 0)  # Orange
                    elif class_names[i] == 'Sad':
                        color = (255, 0, 0)  # Blue
                    
                    cv2.rectangle(frame, 
                                 (chart_x + i * bar_width, chart_y + chart_height - bar_height), 
                                 (chart_x + (i + 1) * bar_width, chart_y + chart_height), 
                                 color, -1)
                    
                    # Add emotion label
                    cv2.putText(frame, class_names[i][:3], 
                               (chart_x + i * bar_width + 2, chart_y + chart_height + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display the emotion label on the frame
                cv2.putText(frame, f'Emotion: {emotion_label}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Display interview status
                status_text = "INTERVIEW IN PROGRESS" if interview_in_progress else "INTERVIEW PAUSED"
                cv2.putText(frame, status_text, 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255) if interview_in_progress else (0, 128, 255), 2)
                
                # Display current question
                if current_question and len(current_question) > 50:
                    display_question = current_question[:50] + "..."
                else:
                    display_question = current_question
                    
                cv2.putText(frame, f"Q{question_number}: {display_question}", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 0, 0), 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global interview_in_progress, emotion_history, current_question, question_number
    global interview_summary, current_interview_id, question_answers, current_candidate
    
    data = request.json
    role = data.get('role', 'general')
    
    # Get candidate info
    current_candidate = {
        "name": data.get('name', ''),
        "email": data.get('email', ''),
        "position": role
    }
    
    # Save candidate to database if email is provided
    if current_candidate["email"]:
        existing_candidate = candidates_collection.find_one({"email": current_candidate["email"]})
        if existing_candidate:
            candidates_collection.update_one(
                {"email": current_candidate["email"]},
                {"$set": {"name": current_candidate["name"], "position": current_candidate["position"]}}
            )
        else:
            candidates_collection.insert_one(current_candidate)
    
    # Create new interview record
    new_interview = {
        "candidate": current_candidate,
        "start_time": datetime.now(),
        "end_time": None,
        "questions": [],
        "summary": None,
        "notes": "",
        "status": "in_progress"
    }
    
    insert_result = interviews_collection.insert_one(new_interview)
    current_interview_id = str(insert_result.inserted_id)
    
    interview_in_progress = True
    emotion_history = []
    interview_summary = ""
    question_number = 1
    question_answers = []
    
    # Get initial question
    if role in interview_questions:
        current_question = interview_questions[role][0]
    else:
        current_question = interview_questions["general"][0]
    
    # Add first question to database
    interviews_collection.update_one(
        {"_id": insert_result.inserted_id},
        {"$push": {"questions": {"number": question_number, "text": current_question, "answer": "", "time": datetime.now()}}}
    )
    
    # Add first question to local tracking
    question_answers.append({
        "number": question_number,
        "question": current_question,
        "answer": "",
        "time": datetime.now()
    })
    
    return jsonify({
        'status': 'started',
        'question': current_question,
        'question_number': question_number,
        'interview_id': current_interview_id
    })


@app.route('/next_question', methods=['POST'])
def next_question():
    global current_question, question_number, question_answers
    
    if not interview_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Interview not in progress'
        })
    
    data = request.json
    role = data.get('role', 'general')
    use_ai = data.get('use_ai', False)
    previous_answer = data.get('answer', '')
    
    # Save the answer to the previous question
    if question_number > 0 and question_number <= len(question_answers):
        # Update local tracking
        question_answers[question_number-1]['answer'] = previous_answer
        
        # Update in MongoDB - using proper error handling
        if current_interview_id:
            try:
                # Convert string ID to ObjectId
                interviews_collection.update_one(
                    {"_id": ObjectId(current_interview_id), "questions.number": question_number},
                    {"$set": {"questions.$.answer": previous_answer}}
                )
                print(f"Updated answer for question {question_number}")
            except Exception as e:
                print(f"Error updating previous answer: {e}")
    
    # Increment question counter
    question_number += 1
    
    # Generate next question
    if use_ai:
        # Use GenAI to generate the next question based on emotions
        previous_questions = [qa['question'] for qa in question_answers]
        current_question = generate_ai_question(role, current_emotion, previous_questions)
    else:
        # Use predefined questions
        if role in interview_questions and question_number <= len(interview_questions[role]):
            current_question = interview_questions[role][question_number - 1]
        else:
            current_question = "No more predefined questions. You can end the interview."
    
    # Add new question to tracking
    question_answers.append({
        "number": question_number,
        "question": current_question,
        "answer": "",
        "time": datetime.now()
    })
    
    # Add to MongoDB
    if current_interview_id:
        try:
            interviews_collection.update_one(
                {"_id": ObjectId(current_interview_id)},
                {"$push": {"questions": {"number": question_number, "text": current_question, "answer": "", "time": datetime.now()}}}
            )
            print(f"Added new question {question_number}")
        except Exception as e:
            print(f"Error adding new question: {e}")
    
    return jsonify({
        'status': 'success',
        'question': current_question,
        'question_number': question_number,
        'current_emotion': current_emotion
    })  
@app.route('/save_answer', methods=['POST'])
def save_answer():
    if not interview_in_progress or not current_interview_id:
        return jsonify({
            'status': 'error',
            'message': 'Interview not in progress'
        })
    
    data = request.json
    answer = data.get('answer', '')
    q_number = data.get('question_number', question_number)
    
    # Update local tracking
    if 0 < q_number <= len(question_answers):
        question_answers[q_number-1]['answer'] = answer
    
    # Update in MongoDB - FIX: Use ObjectId correctly
    try:
        interviews_collection.update_one(
            {"_id": ObjectId(current_interview_id), "questions.number": q_number},
            {"$set": {"questions.$.answer": answer}}
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Answer saved'
        })
    except Exception as e:
        print(f"Error saving answer: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error saving answer: {str(e)}'
        })
        
@app.route('/end_interview', methods=['POST'])
def end_interview():
    global interview_in_progress, interview_summary
    
    if not interview_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Interview not in progress'
        })
    
    # Save final answer if provided
    data = request.json
    final_answer = data.get('final_answer', '')
    if final_answer and question_number > 0 and question_number <= len(question_answers):
        question_answers[question_number-1]['answer'] = final_answer
        
        # Update in MongoDB - FIX: Use ObjectId correctly
        if current_interview_id:
            interviews_collection.update_one(
                {"_id": ObjectId(current_interview_id), "questions.number": question_number},
                {"$set": {"questions.$.answer": final_answer}}
            )
    
    role = data.get('role', current_candidate.get('position', 'general'))
    
    print(f"Generating summary for role: {role}")
    print(f"Emotion history length: {len(emotion_history)}")
    print(f"Q&A pairs: {len(question_answers)}")
    
    # Generate interview summary
    interview_summary = generate_interview_summary(role, emotion_history, question_answers)
    print(f"Generated summary length: {len(interview_summary)}")
    
    
    # Process emotion history
    emotion_counts = {}
    for _, emotion in emotion_history:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    # Update interview record in MongoDB - FIX: Use ObjectId correctly
    if current_interview_id:
        interviews_collection.update_one(
            {"_id": ObjectId(current_interview_id)},
            {"$set": {
                "end_time": datetime.now(),
                "summary": interview_summary,
                "emotion_counts": emotion_counts,
                "status": "completed"
            }}
        )
    
    interview_in_progress = False
    
    return jsonify({
        'status': 'ended',
        'emotion_history': emotion_history,
        'emotion_counts': emotion_counts,
        'summary': interview_summary,  # Make sure this is a string
        'questions': question_answers,  # Include the full Q&A data
        'interview_id': current_interview_id
        })

@app.route('/get_current_state', methods=['GET'])
def get_current_state():
    return jsonify({
        'interview_in_progress': interview_in_progress,
        'current_emotion': current_emotion,
        'question': current_question,
        'question_number': question_number,
        'candidate': current_candidate
    })

@app.route('/get_summary', methods=['GET'])
def get_summary():
    interview_id = request.args.get('interview_id', current_interview_id)
    
    if interview_id:
        # Get from MongoDB - FIX: Use ObjectId correctly
        try:
            interview = interviews_collection.find_one({"_id": ObjectId(interview_id)})
            if interview:
                return jsonify({
                    'summary': interview.get('summary', ''),
                    'emotion_counts': interview.get('emotion_counts', {}),
                    'candidate': interview.get('candidate', {}),
                    'questions': interview.get('questions', []),
                    'notes': interview.get('notes', '')
                })
        except Exception as e:
            print(f"Error retrieving interview: {e}")
    
    if not interview_summary:
        return jsonify({
            'status': 'error',
            'message': 'No interview summary available'
        })
    
    return jsonify({
        'summary': interview_summary,
        'emotion_history': emotion_history
    })

@app.route('/save_notes', methods=['POST'])
def save_notes():
    notes = request.json.get('notes', '')
    interview_id = request.json.get('interview_id', current_interview_id)
    
    # Save to text file as backup
    file_path = f'interview_notes_{interview_id}.txt'
    with open(file_path, 'w') as f:
        f.write(notes)
    
    # Save to MongoDB - FIX: Use ObjectId correctly
    if interview_id:
        try:
            interviews_collection.update_one(
                {"_id": ObjectId(interview_id)},
                {"$set": {"notes": notes}}
            )
            return jsonify({
                'status': 'success',
                'message': 'Notes saved successfully'
            })
        except Exception as e:
            print(f"Error saving notes: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Error saving notes: {str(e)}'
            })
    
    return jsonify({
        'status': 'error',
        'message': 'No interview ID provided'
    })

@app.route('/get_interviews', methods=['GET'])
def get_interviews():
    # Get list of interviews for display
    try:
        interviews = list(interviews_collection.find(
            {}, 
            {"candidate": 1, "start_time": 1, "end_time": 1, "status": 1}
        ).sort("start_time", -1).limit(50))
        
        # Convert ObjectId to string for JSON serialization
        for interview in interviews:
            interview["_id"] = str(interview["_id"])
            if "start_time" in interview:
                interview["start_time"] = interview["start_time"].strftime("%Y-%m-%d %H:%M:%S")
            if "end_time" in interview:
                interview["end_time"] = interview["end_time"].strftime("%Y-%m-%d %H:%M:%S") if interview["end_time"] else None
        
        return jsonify({
            'interviews': interviews
        })
    except Exception as e:
        print(f"Error retrieving interviews: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error retrieving interviews',
            'interviews': []
        })

@app.route('/get_candidates', methods=['GET'])
def get_candidates():
    # Get list of candidates
    try:
        candidates = list(candidates_collection.find({}, {"name": 1, "email": 1, "position": 1}))
        
        # Convert ObjectId to string for JSON serialization
        for candidate in candidates:
            candidate["_id"] = str(candidate["_id"])
        
        return jsonify({
            'candidates': candidates
        })
    except Exception as e:
        print(f"Error retrieving candidates: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error retrieving candidates',
            'candidates': []
        })

@app.route('/get_emotion_data', methods=['GET'])
def get_emotion_data():
    interview_id = request.args.get('interview_id')
    
    if not interview_id:
        return jsonify({
            'status': 'error',
            'message': 'Interview ID is required',
            'data': []
        })
    
    try:
        # Get emotion data for specific interview
        emotion_data = list(emotions_collection.find(
            {"interview_id": interview_id}, 
            {"timestamp": 1, "emotion": 1, "confidence": 1, "all_emotions": 1, "_id": 0}
        ).sort("timestamp", 1))
        
        # Format timestamps for JSON
        for item in emotion_data:
            item["timestamp"] = item["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            'status': 'success',
            'data': emotion_data
        })
    except Exception as e:
        print(f"Error retrieving emotion data: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Error retrieving emotion data',
            'data': []
        })
        
@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    # Get recent interviews
    interviews = list(interviews_collection.find().sort("start_time", -1).limit(10))
    
    # Convert ObjectId to string for each interview
    for interview in interviews:
        interview["_id"] = str(interview["_id"])
        if "start_time" in interview:
            interview["start_time"] = interview["start_time"].strftime("%Y-%m-%d %H:%M:%S")
        if "end_time" in interview and interview["end_time"]:
            interview["end_time"] = interview["end_time"].strftime("%Y-%m-%d %H:%M:%S")
    return render_template('dashboard.html', interviews=interviews)

if __name__ == '__main__':
    app.run(debug=True)