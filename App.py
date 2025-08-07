

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

# --- Add global custom CSS for modern look ---
st.markdown('''
    <style>
    body {
        background-color: #f4f6fb;
    }
    .main {
        background-color: #f4f6fb;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .section-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(60,60,60,0.08);
        padding: 2rem 2.5rem 1.5rem 2.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid #e6e6e6;
    }
    .section-title {
        font-size: 1.5em;
        font-weight: 700;
        color: #2d3a4a;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }
    .chip {
        display: inline-block;
        padding: 0.25em 0.8em;
        margin: 0.15em;
        border-radius: 1em;
        background: #e0e7ff;
        color: #1e293b;
        font-size: 0.95em;
    }
    .chip-missing {
        background: #ffe0e0;
        color: #b71c1c;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3em 0.8em;
        border-radius: 1em;
        font-weight: bold;
        color: white;
        background: #4caf50;
        margin-left: 1em;
    }
    .divider {
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0 1.5rem 0;
    }
    </style>
''', unsafe_allow_html=True)

# --- Improved Title and Description ---
st.markdown('<div class="section-card" style="text-align:center;"><span class="section-title">📄 AI Resume Analyzer & Skill Enhancer</span><br><span style="font-size:1.1em; color:#4b5563;">Upload your Resume and Job Description to analyze missing skills and get YouTube course recommendations!</span></div>', unsafe_allow_html=True)

# --- Upload Section ---
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<span class="section-title">⬆️ Upload Files</span>', unsafe_allow_html=True)
resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
st.markdown('</div>', unsafe_allow_html=True)

if resume_file and job_file:
    # --- Summary Section ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">📝 Document Summaries</span>', unsafe_allow_html=True)
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**📌 Resume Summary**')
        st.write(generate_summary(resume_text))
    with col2:
        st.markdown('**📌 Job Description Summary**')
        st.write(generate_summary(job_text))
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Analysis Section ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="section-title">🔍 Skill Analysis & Matching</span>', unsafe_allow_html=True)
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
        st.markdown('<div style="margin-bottom:1.5rem;">', unsafe_allow_html=True)
        st.markdown(f'<span style="font-size:1.2em; font-weight:600;">🎯 Matching Score: <span class="score-badge">{st.session_state.matching_score}%</span></span>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('**📄 Resume Skills**')
            st.markdown(' '.join([f'<span class="chip">{skill}</span>' for skill in st.session_state.resume_skills]) or '<i>No skills found</i>', unsafe_allow_html=True)
        with col2:
            st.markdown('**💼 Job Required Skills**')
            st.markdown(' '.join([f'<span class="chip">{skill}</span>' for skill in st.session_state.job_skills]) or '<i>No skills found</i>', unsafe_allow_html=True)
        with col3:
            st.markdown('**⚠️ Missing Skills**')
            st.markdown(' '.join([f'<span class="chip chip-missing">{skill}</span>' for skill in st.session_state.missing_skills]) or '<i>No missing skills</i>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        plot_skill_distribution_pie(st.session_state.resume_skills, st.session_state.job_skills)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("📚 Get Recommended Courses"):
            all_courses = []
            for skill in st.session_state.missing_skills:
                all_courses.extend(fetch_youtube_courses(skill))
            df = pd.DataFrame(all_courses)
            st.table(df if not df.empty else "No courses found.")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

# --- Saved Analyses Section ---
st.sidebar.header("View Saved Analyses")
user_filter = st.sidebar.text_input("Filter by user name")
if st.sidebar.button("Show Analyses"):
    conn = sqlite3.connect('analysis_results.db')
    if user_filter:
        df = pd.read_sql_query("SELECT * FROM analysis WHERE user_name = ?", conn, params=(user_filter,))
    else:
        df = pd.read_sql_query("SELECT * FROM analysis", conn)
    conn.close()
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if not df.empty:
        st.markdown(f"<span class='section-title'>📊 Saved Analyses{' for ' + user_filter if user_filter else ''}</span>", unsafe_allow_html=True)
        for index, row in df.iterrows():
            score = row['matching_score']
            if score >= 80:
                badge_color = "#4caf50"
                badge_text = "Excellent"
            elif score >= 60:
                badge_color = "#2196f3"
                badge_text = "Good"
            elif score >= 40:
                badge_color = "#ff9800"
                badge_text = "Fair"
            else:
                badge_color = "#f44336"
                badge_text = "Poor"
            st.markdown(f"""
            <div style='background:#f9f9fa; border-radius:12px; box-shadow:0 2px 8px rgba(60,60,60,0.08); padding:1.5rem 2rem; margin-bottom:2rem; border:1px solid #e6e6e6;'>
                <div style='font-size:1.2em; font-weight:700;'>
                    👤 {row['user_name']}
                    <span class='score-badge' style='background:{badge_color};'>{badge_text} ({score}%)</span>
                </div>
                <div class='section-title' style='font-size:1.1em;'>📄 Resume Skills</div>
                {''.join([f'<span class="chip">{skill.strip()}</span>' for skill in row['resume_skills'].split(',') if skill.strip()]) or '<i>No skills found</i>'}
                <div class='section-title' style='font-size:1.1em;'>💼 Job Required Skills</div>
                {''.join([f'<span class="chip">{skill.strip()}</span>' for skill in row['job_skills'].split(',') if skill.strip()]) or '<i>No skills found</i>'}
                <div class='section-title' style='font-size:1.1em;'>⚠️ Missing Skills</div>
                {''.join([f'<span class="chip chip-missing">{skill.strip()}</span>' for skill in row['missing_skills'].split(',') if skill.strip()]) or '<i>No missing skills</i>'}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No saved analyses found.")
    st.markdown('</div>', unsafe_allow_html=True)
