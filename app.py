import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    """Load the trained baseline model"""
    model_path = 'baseline_model.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"⚠️ Error loading model: {str(e)}")
            return None
    else:
        st.error("⚠️ Model file 'baseline_model.pkl' not found!")
        return None

model = load_model()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

def encode_categorical_features(data):
    """Encode categorical features to match the trained model format"""
    ethnicity_map = {'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3}
    education_map = {'High School': 0, 'Some College': 1, 'Bachelor': 2, 'Higher': 3}
    support_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    return {
        'Age': data['Age'],
        'Gender': 1 if data['Gender'] == 'Male' else 0,
        'Ethnicity': ethnicity_map.get(data['Ethnicity'], 0),
        'ParentalEducation': education_map.get(data['ParentalEducation'], 2),
        'StudyTimeWeekly': data['StudyTimeWeekly'],
        'Absences': data['Absences'],
        'Tutoring': 1 if data['Tutoring'] == 'Yes' else 0,
        'ParentalSupport': support_map.get(data['ParentalSupport'], 1),
        'Extracurricular': 1 if data['Extracurricular'] else 0,
        'Sports': 1 if data['Sports'] else 0,
        'Music': 1 if data['Music'] else 0,
        'Volunteering': 1 if data['Volunteering'] else 0
    }

def calculate_prediction(data):
    """Calculate GPA prediction using the trained baseline model"""
    if model is None:
        st.error("⚠️ Model not available. Please ensure baseline_model.pkl exists.")
        return None
    
    try:
        encoded_data = encode_categorical_features(data)
        
        feature_order = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 
                        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 
                        'Sports', 'Music', 'Volunteering']
        
        feature_df = pd.DataFrame([encoded_data], columns=feature_order)
        predicted_gpa = model.predict(feature_df)[0]
        predicted_gpa = max(0.0, min(4.0, predicted_gpa))
        grade_percentage = (predicted_gpa / 4.0) * 100
        
    except Exception as e:
        st.error(f"⚠️ Model prediction failed: {str(e)}")
        return None
    
    # Determine performance level
    if predicted_gpa >= 3.7:
        level, icon = 'Excellent', '🏆'
    elif predicted_gpa >= 3.0:
        level, icon = 'Good', '✅'
    elif predicted_gpa >= 2.5:
        level, icon = 'Average', '🎯'
    elif predicted_gpa >= 2.0:
        level, icon = 'Below Average', '📈'
    else:
        level, icon = 'At Risk', '⚠️'
    
    return {
        'gpa': predicted_gpa,
        'grade_percentage': grade_percentage,
        'level': level,
        'icon': icon
    }

def get_recommendations(data, prediction):
    """Generate personalized recommendations"""
    recs = []
    
    if data['StudyTimeWeekly'] < 15:
        recs.append("Increase weekly study time to at least 15-20 hours for better performance")
    
    if data['Absences'] > 10:
        recs.append("Improve attendance - high absences significantly impact learning")
    
    if data['Tutoring'] == 'No' and prediction['gpa'] < 3.0:
        recs.append("Consider tutoring support to strengthen understanding in challenging subjects")
    
    if data['ParentalSupport'] == 'Low':
        recs.append("Seek additional support from teachers or mentors to supplement parental involvement")
    
    activity_count = sum([data['Extracurricular'], data['Sports'], data['Music'], data['Volunteering']])
    
    if activity_count == 0:
        recs.append("Join extracurricular activities to develop well-rounded skills and boost college applications")
    
    if data['StudyTimeWeekly'] >= 15 and data['Absences'] <= 5 and prediction['gpa'] >= 3.5:
        recs.append("Maintain current study habits and involvement - you're on an excellent trajectory")
        recs.append("Consider taking on leadership roles in activities to further develop skills")
    
    if not recs:
        recs.append("Continue current study habits and maintain consistent effort")
        recs.append("Set challenging academic goals to continue growing")
    
    return recs

# Header
st.title("🎓 Student Performance Predictor")
st.markdown("Enter student data to predict final performance and receive personalized recommendations for improvement")

# Two Column Layout
col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN - Input Form
with col1:
    st.subheader("👤 Student Information")
    
    student_name = st.text_input("Student Name", placeholder="Enter student name")
    
    st.divider()
    
    age = st.slider("📅 Age", min_value=15, max_value=50,value=16)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
    parental_education = st.selectbox("Parental Education Level", 
                                     ["High School", "Some College", "Bachelor", "Higher"], 
                                     index=2)
    
    st.divider()
    st.subheader("📊 Academic Metrics")
    
    study_time = st.slider("📚 Study Time Weekly (hours)", min_value=0, max_value=40, value=10)
    st.caption(f"**{study_time} hours** per week")
    
    absences = st.slider("📅 Absences (days)", min_value=0, max_value=30, value=5)
    st.caption(f"**{absences} days** absent")
    
    st.divider()
    st.subheader("🎯 Support & Activities")
    
    col_a, col_b = st.columns(2)
    with col_a:
        tutoring = st.radio("Tutoring", ["Yes", "No"], index=1)
    with col_b:
        parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"], index=1)
    
    st.subheader("Extracurricular Activities")
    
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        extracurricular = st.checkbox("Extracurricular")
        sports = st.checkbox("Sports")
    with col_act2:
        music = st.checkbox("Music")
        volunteering = st.checkbox("Volunteering")
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 Reset", type="secondary", use_container_width=True)

# RIGHT COLUMN - Results
with col2:
    st.subheader("📊 Prediction Results")
    
    if reset_btn:
        st.session_state.prediction = None
    
    if predict_btn:
        if not student_name.strip():
            st.error("⚠️ Please enter student name")
        else:
            form_data = {
                'Name': student_name,
                'Age': age,
                'Gender': gender,
                'Ethnicity': ethnicity,
                'ParentalEducation': parental_education,
                'StudyTimeWeekly': study_time,
                'Absences': absences,
                'Tutoring': tutoring,
                'ParentalSupport': parental_support,
                'Extracurricular': extracurricular,
                'Sports': sports,
                'Music': music,
                'Volunteering': volunteering
            }
            
            prediction = calculate_prediction(form_data)
            st.session_state.prediction = prediction
    
    # Display Results
    if st.session_state.prediction:
        pred = st.session_state.prediction
        
        st.markdown(f"**{student_name}**")
        st.divider()
        
        # Performance Badge using native Streamlit containers
        if pred['level'] == 'Excellent':
            container = st.success
        elif pred['level'] == 'Good':
            container = st.info
        elif pred['level'] == 'Average':
            container = st.warning
        else:
            container = st.error
        
        with container(f"{pred['icon']} **Performance Level: {pred['level']}**"):
            col_perf1, col_perf2 = st.columns(2)
            with col_perf1:
                st.metric("Predicted GPA", f"{pred['gpa']:.2f}")
            with col_perf2:
                st.metric("Predicted Grade", f"{pred['grade_percentage']:.0f}%")
        
        # Radar Chart
        with st.expander("📊 Performance Metrics", expanded=True):
            activity_count = sum([extracurricular, sports, music, volunteering])
            support_val = {'Low': 33, 'Medium': 66, 'High': 100}.get(parental_support, 66)
            
            radar_data = pd.DataFrame({
                'metric': ['Current Grade', 'Attendance', 'Study Hours', 'Participation'],
                'value': [
                    pred['grade_percentage'],
                    ((30 - absences) / 30) * 100,
                    (study_time / 40) * 100,
                    support_val + (activity_count / 4) * 20
                ]
            })
            
            radar_data['value'] = radar_data['value'].clip(0, 100)
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_data['value'],
                theta=radar_data['metric'],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.5)',
                line=dict(color='#3b82f6', width=2),
                name='Performance Score'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    angularaxis=dict(tickfont=dict(size=12))
                ),
                showlegend=False,
                height=300,
                margin=dict(l=60, r=60, t=20, b=60)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Recommendations
        st.subheader("📈 Recommendations")
        
        current_data = {
            'StudyTimeWeekly': study_time,
            'Absences': absences,
            'Tutoring': tutoring,
            'ParentalSupport': parental_support,
            'Extracurricular': extracurricular,
            'Sports': sports,
            'Music': music,
            'Volunteering': volunteering
        }
        
        recommendations = get_recommendations(current_data, pred)
        
        for idx, rec in enumerate(recommendations, 1):
            st.info(f"**{idx}.** {rec}", icon="💡")
    
    else:
        st.info("🎓 Enter student data to see predictions")