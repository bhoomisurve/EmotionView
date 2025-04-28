# EmotionView 🎥🧠  
**AI-Powered Emotional Interview Analysis System**

> EmotionView transforms traditional interviews by adding real-time emotional intelligence, AI-driven question generation, and deep analytics — all in a sleek, browser-based interface.

---

## Overview
EmotionView is designed for HR teams, hiring managers, and recruiters who want to go beyond canned questions and gut feelings. By combining computer vision, deep learning, and generative AI, EmotionView:

- **Detects candidate emotions** continuously via webcam using a proven TensorFlow model.  
- **Adapts questions on the fly** with Google Gemini Pro, tailoring each follow-up to the candidate’s current mood.  
- **Logs every response and emotion** timestamp-by-timestamp, storing it securely in MongoDB Atlas.  
- **Generates a professional interview summary**, complete with stress-management insights, emotional stability metrics, and communication quality.  
- **Provides a searchable dashboard** for reviewing past sessions, filtering by date, position, or emotion trends.

Experience faster, fairer, and more insightful hiring interviews — powered by EmotionView.

---

## Features
- 🎥 **Live Video Streaming & Emotion Detection**  
  - Face detection with OpenCV’s Haar cascades  
  - Emotion classification into: Angry, Disgusted, Fear, Happy, Sad, Surprise, Neutral  

- 🤖 **AI-Generated Interview Questions**  
  - Prompts Google Gemini Pro with candidate’s emotional state  
  - Falls back to predefined questions when AI is unavailable  

- 📊 **Interview Analytics & Dashboard**  
  - Emotion timeline visualization with Chart.js  
  - Aggregate emotion distribution charts  
  - Search, filter, and export interview records  

- ☁️ **Cloud-Ready Storage**  
  - MongoDB Atlas for candidate, interview, and emotion data  
  - Secure environment variable configuration via `.env`  

- 🔒 **Security & Privacy**  
  - No keys or URIs hardcoded in code  
  - Notes export to text/PDF for offline archiving  

---

Usage

  1. Start Interview

        Enter candidate name, email, and position.

        Click Start Interview to launch the video feed and begin emotion tracking.

   2.  Conduct Q&A

        Save each answer manually or generate the next question with AI.

        Monitor real-time emotion badges and timeline.

   3.  End Interview

        Click End Interview to trigger AI-generated summary and persist data.

        Review or export the summary and notes.

  4.  Dashboard

        Navigate to Dashboard to view all interviews.

        Filter by date, position, or emotion.

        Export CSV/PDF or drill down into emotion analytics.

Tech Stack

    Backend: Python 3, Flask, OpenCV, TensorFlow

    AI Integration: Google Generative AI (Gemini Pro)

    Database: MongoDB Atlas (via PyMongo)

    Frontend: HTML5, TailwindCSS, Chart.js, vanilla JavaScript

  
Future Enhancements

    📄 One-click PDF reports on interview completion

    ✉️ Automated email delivery of summaries

    🌐 Multi-language support for global teams

    🔐 Role-based access control for secure admin/user separation


**Author**

Bhoomika Surve
