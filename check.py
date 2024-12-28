import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# Loading the saved models
diabetes_model = pickle.load(open('/workspaces/predicition/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open("/workspaces/predicition/heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open("/workspaces/predicition/parkinsons_model.sav", 'rb'))

# Page Configuration
st.set_page_config(page_title="Disease Prediction System", layout="wide")
st.markdown("""<style>
    .reportview-container {
        background: #fab9b9;
    }
    .sidebar .sidebar-content {
        background: #fff;
    }
</style>""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Helper function to display warnings
def validate_inputs(inputs):
    try:
        return [float(value) for value in inputs]
    except ValueError:
        st.error("Please ensure all inputs are numeric!")
        return None

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    st.write("Predict if a person is diabetic based on health metrics.")

    # Input fields
    cols = st.columns(2)
    Pregnancies = cols[0].text_input('Number of Pregnancies')
    Glucose = cols[0].text_input('Glucose Level')
    BloodPressure = cols[0].text_input('Blood Pressure value')
    SkinThickness = cols[0].text_input('Skin Thickness value')

    Insulin = cols[1].text_input('Insulin Level')
    BMI = cols[1].text_input('BMI value')
    DiabetesPedigreeFunction = cols[1].text_input('Diabetes Pedigree Function value')
    Age = cols[1].text_input('Age of the Person')

    diab_diagnosis = ''

    # Prediction
    if st.button('Get Diabetes Test Result'):
        inputs = validate_inputs([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if inputs:
            diab_prediction = diabetes_model.predict([inputs])
            diab_diagnosis = 'The person is Diabetic' if diab_prediction[0] == 1 else 'The person is Not Diabetic'
            st.success(diab_diagnosis)

    # Visualization
    if st.button('Visualize Diabetes Data'):
        inputs = validate_inputs([Pregnancies, Glucose, BloodPressure, SkinThickness])
        if inputs:
            labels = ['Pregnancies', 'Glucose', 'BP', 'Skin Thickness']
            fig, ax = plt.subplots()
            ax.bar(labels, inputs, color='skyblue')
            ax.set_title("Diabetes Input Visualization")
            st.pyplot(fig)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    st.write("Predict if a person has heart disease based on health parameters.")

    # Input fields
    cols = st.columns(2)
    age = cols[0].number_input('Age of the Person', min_value=0, step=1)
    sex = cols[0].selectbox('Sex', ['Male', 'Female'])
    cp = cols[0].number_input('Chest Pain Types (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)', min_value=0, step=1)
    trestbps = cols[0].number_input('Resting Blood Pressure', min_value=0)
    chol = cols[1].number_input('Serum Cholestoral in mg/dl', min_value=0)
    fbs = cols[1].number_input('Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)', min_value=0, max_value=1, step=1)
    restecg = cols[1].number_input('Resting Electrocardiographic Results (0: Normal, 1: Having ST-T wave abnormality, 2: Left ventricular hypertrophy)', min_value=0, step=1)
    thalach = cols[1].number_input('Maximum Heart Rate Achieved', min_value=0)
    exang = cols[1].number_input('Exercise Induced Angina (1: Yes, 0: No)', min_value=0, max_value=1, step=1)
    oldpeak = cols[1].number_input('ST depression induced by exercise')
    slope = cols[0].number_input('Slope of the Peak Exercise ST Segment', min_value=0, step=1)  # Adding the slope feature
    ca = cols[0].number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, step=1)  # Adding the ca feature
    thal = cols[1].number_input('Thalassemia (3: Normal, 6: Fixed defect, 7: Reversible defect)', min_value=0, max_value=3, step=1)  # Adding the thal feature

    heart_diagnosis = ''

    # Prediction
    if st.button('Heart Test Result'):
        # Convert 'sex' to 1 or 0 (Male: 1, Female: 0)
        sex = 1 if sex == 'Male' else 0
        
        # Prepare input features (now with all 13 features)
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,slope, ca, thal]
        
        # Ensure you have all 13 features and pass them in the correct order
        heart_prediction = heart_disease_model.predict([features])
        
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is suffering from Heart disease'
        else:
            heart_diagnosis = 'The person is Not suffering from Heart disease'

        st.success(heart_diagnosis)

    # Visualization
    if st.button('Visualize Heart Data'):
        inputs = validate_inputs([cp, trestbps, chol, fbs, restecg, thalach, exang, slope, ca, thal])
        if inputs:
            labels = ['cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal',]
            fig, ax = plt.subplots()
            ax.bar(labels, inputs, color='Red')
            ax.set_title("Heart Input Visualization")
            st.pyplot(fig)


# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Prediction")
    st.write("Predict if a person has Parkinson's disease based on health metrics.\n")

    cols = st.columns(4)
    fo = cols[0].text_input('MDVP:Fo(Hz)')
    fhi = cols[0].text_input('MDVP:Fhi(Hz)')
    flo = cols[0].text_input('MDVP:Flo(Hz)')
    Jitter_percent = cols[0].text_input('MDVP:Jitter(%)')
    Jitter_Abs = cols[0].text_input('MDVP:Jitter(Abs)')
    RAP = cols[0].text_input('MDVP:RAP')
    PPQ = cols[1].text_input('MDVP:PPQ')
    DDP = cols[1].text_input('Jitter:DDP')
    Shimmer = cols[1].text_input('MDVP:Shimmer')
    Shimmer_dB = cols[1].text_input('MDVP:Shimmer(dB)')
    APQ3 = cols[1].text_input('Shimmer:APQ3')
    APQ5 = cols[2].text_input('Shimmer:APQ5')
    APQ = cols[2].text_input('MDVP:APQ')
    DDA = cols[2].text_input('Shimmer:DDA')
    NHR = cols[2].text_input('NHR')
    HNR = cols[2].text_input('HNR')
    RPDE = cols[2].text_input('RPDE')
    DFA = cols[3].text_input('DFA')
    spread1 = cols[3].text_input('spread1')
    spread2 = cols[3].text_input('spread2')
    D2 = cols[3].text_input('D2')
    PPE = cols[3].text_input('PPE')
    
    parkinsons_diagnosis = ''

    # Prediction
    if st.button('Parkinsons Test Result'):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        
        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = 'The person is suffering from Parkinsons disease'
        else:
            parkinsons_diagnosis = 'The person is Not suffering from Parkinsons disease'

    # Visualization
    if st.button("Visualize Parkinson's Data"):
        inputs = validate_inputs([fo, fhi, flo, Jitter_percent, Shimmer])
        if inputs:
            labels = ['Fo (Hz)', 'Fhi (Hz)', 'Flo (Hz)', 'Jitter (%)', 'Shimmer']
            fig, ax = plt.subplots()
            ax.bar(labels, inputs, color='lightgreen')
            ax.set_title("Parkinson's Input Visualization")
            st.pyplot(fig)
