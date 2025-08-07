

import streamlit as st
import requests
import pandas as pd
import pdfplumber
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import googleapiclient.discovery
import spacy
from spacy.cli import download
import sqlite3

# Load NLP model
model_name = "en_core_web_md"
try:
    nlp = spacy.load(model_name)
except OSError:
    download(model_name)
    nlp = spacy.load(model_name)

# Load AI Model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# YouTube API Key (Replace with a new secured key)
YOUTUBE_API_KEY = "AIzaSyBoRgw0WE_KzTVNUvH8d4MiTo1zZ2SqKPI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Initialize session state
if "skills_analyzed" not in st.session_state:
    st.session_state.skills_analyzed = False
    st.session_state.show_courses = False
    st.session_state.missing_skills = []
    st.session_state.matching_score = 0.0
    st.session_state.resume_skills = []
    st.session_state.job_skills = []

def init_db():
    conn = sqlite3.connect('analysis_results.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            resume_skills TEXT,
            job_skills TEXT,
            missing_skills TEXT,
            matching_score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Function to fetch courses from YouTube
def fetch_youtube_courses(skill):
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=f"{skill} course", part="snippet", maxResults=5, type="video")
    response = request.execute()
    
    courses = []
    for item in response.get("items", []):
        if (
            "id" in item and
            "videoId" in item["id"] and
            "snippet" in item and
            "title" in item["snippet"] and
            "channelTitle" in item["snippet"]
        ):
            courses.append({
                "Title": item["snippet"]["title"],
                "Channel": item["snippet"]["channelTitle"],
                "Video Link": f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'
            })
    return courses

# Function to extract text from files
def extract_text(uploaded_file):
    if uploaded_file is not None:
        try:
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "pdf":
                with pdfplumber.open(uploaded_file) as pdf:
                    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()]) or "No text extracted."
            elif ext in ["docx", "doc"]:
                return docx2txt.process(uploaded_file) or "No text extracted."
            elif ext == "txt":
                return uploaded_file.read().decode("utf-8") or "No text extracted."
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    return "No text extracted."

# Function to generate short descriptions
def generate_summary(text):
    sentences = text.split(". ")[:3]
    return "... ".join(sentences) + "..." if sentences else "No content extracted."

# Function to extract skills using NER
def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":  # You can customize this based on your needs
            skills.add(ent.text)
    return list(skills)

# Function to calculate similarity score
def calculate_matching_score(resume_text, job_text):
    embeddings = st_model.encode([resume_text, job_text], convert_to_tensor=True)
    return round(float(util.pytorch_cos_sim(embeddings[0], embeddings[1])[0]), 2) * 100

# Function to plot skill comparison
def plot_skill_distribution_pie(resume_skills, job_skills):
    resume_labels = list(resume_skills) if resume_skills else ["No Skills Found"]
    resume_sizes = [1] * len(resume_skills) if resume_skills else [1]
    
    job_labels = list(job_skills) if job_skills else ["No Skills Found"]
    job_sizes = [1] * len(job_skills) if job_skills else [1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].pie(resume_sizes, labels=resume_labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("Resume Skills Distribution")
    axes[1].pie(job_sizes, labels=job_labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Job Required Skills Distribution")
    st.pyplot(fig)

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.write("Upload your Resume and Job Description to analyze missing skills and get YouTube course recommendations!")

resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)
    
    st.subheader("📌 Resume Summary")
    st.write(generate_summary(resume_text))
    
    st.subheader("📌 Job Description Summary")
    st.write(generate_summary(job_text))
    
    if st.button("Analyze Skills & Matching Score"):
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_text)
        missing_skills = list(set(job_skills) - set(resume_skills))

        st.session_state.skills_analyzed = True
        st.session_state.resume_skills = resume_skills
        st.session_state.job_skills = job_skills
        st.session_state.missing_skills = missing_skills
        st.session_state.matching_score = calculate_matching_score(resume_text, job_text)

    if st.session_state.skills_analyzed:
        st.subheader("🔍 Extracted Skills")
        st.write(f"**Resume Skills:** {', '.join(st.session_state.resume_skills)}")
        st.write(f"**Job Required Skills:** {', '.join(st.session_state.job_skills)}")

        st.subheader("📊 Resume Matching Score")
        st.write(f"Your resume matches **{st.session_state.matching_score}%** of the job requirements.")

        st.subheader("⚠️ Missing Skills")
        if st.session_state.missing_skills:
            st.write(f"You are missing: {', '.join(st.session_state.missing_skills)}")
        
        plot_skill_distribution_pie(st.session_state.resume_skills, st.session_state.job_skills)

        if st.button("📚 Get Recommended Courses"):
            all_courses = []
            for skill in st.session_state.missing_skills:
                all_courses.extend(fetch_youtube_courses(skill))
            df = pd.DataFrame(all_courses)
            st.table(df if not df.empty else "No courses found.")

        # --- Save Analysis Section ---
        with st.form("save_analysis_form"):
            user_name = st.text_input("Enter your name to save this analysis")
            save_btn = st.form_submit_button("Save Analysis")
            if save_btn and user_name:
                conn = sqlite3.connect('analysis_results.db')
                c = conn.cursor()
                c.execute('''
                    INSERT INTO analysis (user_name, resume_skills, job_skills, missing_skills, matching_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    user_name,
                    ','.join(st.session_state.resume_skills),
                    ','.join(st.session_state.job_skills),
                    ','.join(st.session_state.missing_skills),
                    st.session_state.matching_score
                ))
                conn.commit()
                conn.close()
                st.success("Analysis saved successfully!")

st.sidebar.header("View Saved Analyses")
user_filter = st.sidebar.text_input("Filter by user name")
if st.sidebar.button("Show Analyses"):
    conn = sqlite3.connect('analysis_results.db')
    if user_filter:
        df = pd.read_sql_query("SELECT * FROM analysis WHERE user_name = ?", conn, params=(user_filter,))
    else:
        df = pd.read_sql_query("SELECT * FROM analysis", conn)
    conn.close()
    
    if not df.empty:
        st.write(f"### 📊 Saved Analyses{' for ' + user_filter if user_filter else ''}")
        
        for index, row in df.iterrows():
            with st.expander(f"📋 Analysis #{row['id']} - {row['user_name']} (Score: {row['matching_score']}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**👤 User:**", row['user_name'])
                    st.write("**🎯 Matching Score:**", f"{row['matching_score']}%")
                    
                    # Parse skills from stored strings
                    resume_skills = row['resume_skills'].split(',') if row['resume_skills'] else []
                    job_skills = row['job_skills'].split(',') if row['job_skills'] else []
                    missing_skills = row['missing_skills'].split(',') if row['missing_skills'] else []
                    
                    st.write("**📄 Resume Skills:**")
                    if resume_skills:
                        for skill in resume_skills:
                            st.write(f"• {skill.strip()}")
                    else:
                        st.write("No skills found")
                
                with col2:
                    st.write("**💼 Job Required Skills:**")
                    if job_skills:
                        for skill in job_skills:
                            st.write(f"• {skill.strip()}")
                    else:
                        st.write("No skills found")
                    
                    st.write("**⚠️ Missing Skills:**")
                    if missing_skills:
                        for skill in missing_skills:
                            st.write(f"• {skill.strip()}")
                    else:
                        st.write("No missing skills")
                
                # Color-coded score indicator
                score = row['matching_score']
                if score >= 80:
                    st.success(f"🎉 Excellent Match! ({score}%)")
                elif score >= 60:
                    st.info(f"👍 Good Match ({score}%)")
                elif score >= 40:
                    st.warning(f"⚠️ Fair Match ({score}%)")
                else:
                    st.error(f"❌ Poor Match ({score}%)")
    else:
        st.info("No saved analyses found.")
