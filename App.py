
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
import sqlalchemy as db
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

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

# Function to fetch courses from YouTube
def fetch_youtube_courses(skill):
    youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(q=f"{skill} course", part="snippet", maxResults=5, type="video")
    response = request.execute()
    
    return [
        {"Title": item["snippet"]["title"], "Channel": item["snippet"]["channelTitle"], "Video Link": f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'}
        for item in response["items"]
    ]

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

# Database setup
Base = declarative_base()
class Analysis(Base):
    __tablename__ = 'analyses'
    id = Column(Integer, primary_key=True)
    resume_skills = Column(String)
    job_skills = Column(String)
    missing_skills = Column(String)
    matching_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def get_engine():
    return create_engine('sqlite:///analyses.db', connect_args={"check_same_thread": False})

def get_session():
    engine = get_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def save_analysis(resume_skills, job_skills, missing_skills, matching_score):
    session = get_session()
    analysis = Analysis(
        resume_skills=", ".join(resume_skills),
        job_skills=", ".join(job_skills),
        missing_skills=", ".join(missing_skills),
        matching_score=matching_score
    )
    session.add(analysis)
    session.commit()
    session.close()

# Function to fetch all analyses from the database

def fetch_all_analyses():
    session = get_session()
    analyses = session.query(Analysis).order_by(Analysis.timestamp.desc()).all()
    session.close()
    return analyses

# Helper to load Lottie animation from URL

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Analyze Resume", "Saved Analyses"])

# Lottie animation for header
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_ktwnwv5m.json"  # Example student/job search animation
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=180, key="header_anim")

st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .stTable, .stDataFrame {
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 2px 8px #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

if page == "Home":
    st.title("📄 AI Resume Analyzer & Skill Enhancer")
    st.write("Welcome! This app helps students and job seekers analyze their resumes, discover missing skills, and get personalized job and course recommendations. Upload your resume and a job description to get started!")
    st.info("Tip: Use the sidebar to navigate between features.")
    st.image("https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif", caption="Level up your career!", use_column_width=True)

elif page == "Analyze Resume":
    st.header("🔍 Analyze Your Resume vs Job Description")
    resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], help="Upload your resume file here.")
    job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], help="Upload the job description file here.")
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
            save_analysis(resume_skills, job_skills, missing_skills, st.session_state.matching_score)
        if st.session_state.get("skills_analyzed", False):
            st.subheader("🔍 Extracted Skills")
            st.write(f"**Resume Skills:** {', '.join(st.session_state.resume_skills)}")
            st.write(f"**Job Required Skills:** {', '.join(st.session_state.job_skills)}")
            st.subheader("📊 Resume Matching Score")
            st.success(f"Your resume matches **{st.session_state.matching_score}%** of the job requirements.")
            st.subheader("⚠️ Missing Skills")
            if st.session_state.missing_skills:
                st.warning(f"You are missing: {', '.join(st.session_state.missing_skills)}")
            # 3D Pie chart with Plotly
            fig = go.Figure()
            fig.add_trace(go.Pie(labels=st.session_state.resume_skills or ["No Skills"],
                                 values=[1]*len(st.session_state.resume_skills) or [1],
                                 name="Resume Skills",
                                 hole=0.3))
            fig.add_trace(go.Pie(labels=st.session_state.job_skills or ["No Skills"],
                                 values=[1]*len(st.session_state.job_skills) or [1],
                                 name="Job Skills",
                                 hole=0.3))
            fig.update_layout(title_text="Skill Distribution (Resume vs Job)",
                              grid={'rows': 1, 'columns': 2},
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            if st.button("📚 Get Recommended Courses"):
                all_courses = []
                for skill in st.session_state.missing_skills:
                    all_courses.extend(fetch_youtube_courses(skill))
                df = pd.DataFrame(all_courses)
                st.table(df if not df.empty else "No courses found.")

elif page == "Saved Analyses":
    st.header("📂 Saved Analyses")
    analyses = fetch_all_analyses()
    if analyses:
        df = pd.DataFrame([
            {
                "Date": a.timestamp.strftime("%Y-%m-%d %H:%M"),
                "Resume Skills": a.resume_skills,
                "Job Skills": a.job_skills,
                "Missing Skills": a.missing_skills,
                "Matching Score": a.matching_score
            }
            for a in analyses
        ])
        st.dataframe(df)
        st.info("Tip: You can copy or download your results for future reference!")
    else:
        st.info("No saved analyses found. Start by analyzing your resume!")
