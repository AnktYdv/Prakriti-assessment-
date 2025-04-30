import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load model and encoders
try:
    model = joblib.load('prakriti_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoder = joblib.load('target_encoder.pkl')
except FileNotFoundError:
    st.error("Model or encoder files not found. Please run train_model.py first.")
    st.stop()

# Define question options (Q1–Q59)
question_options = {
    # Q1–Q12: Personal Information, Weight, Height (non-predictive)
    "Q1": None,  # First Name (text)
    "Q2": ["Male", "Female"],
    "Q3": None,  # Permanent Address (text)
    "Q4": None,  # Phone Number (text)
    "Q5": None,  # Email Address (text)
    "Q6": None,  # Code No (text)
    "Q7": None,  # Age (text)
    "Q8": None,  # Date (text)
    "Q9": None,  # Interviewer's Name (text)
    "Q10": ["Yes", "No"],
    "Q11": None,  # Weight (text)
    "Q12": None,  # Height (text)
    # Q13–Q59: Prakriti Screening (predictive)
    "Q13": ["Narrow", "Medium", "Wide"],
    "Q14": ["Weakly developed", "Moderately developed", "Well developed"],
    "Q15": ["Thin Musculature", "Soft and Loosely knitted Musculature", "Smooth and Firmly knitted Musculature"],
    "Q16": ["Small", "Medium", "Large"],
    "Q17": ["Smooth", "Soft", "Rough"],
    "Q18": ["Reddish", "Pale", "Pink"],
    "Q19": ["Small", "Medium", "Large"],
    "Q20": ["Cracked", "Lustrous", "Rough", "Moles", "Marks", "Pimples", "Freckles", "Wrinkles", "None"],
    "Q21": ["Fair with reddish tinge", "Fair with yellowish tinge", "Dark", "Dusky", "Wheatish", "Fair with pale tinge", "Fair with pink tinge"],
    "Q22": ["Dry", "Oily", "Normal", "Seasonal"],
    "Q23": ["Thick", "Thin"],
    "Q24": ["Thick", "Thin"],
    "Q25": ["Graying", "Falling", "Breaking", "Split at ends", "None"],
    "Q26": ["Dry", "Oily", "Normal", "Seasonal"],
    "Q27": ["Nails Brittle", "Palm Cracked", "Sole Cracked", "Lips Cracked"],
    "Q28": ["Regular", "Irregular", "Occasionally Irregular"],
    "Q29": ["Frequent", "Infrequent"],
    "Q30": ["Sweet", "Sour", "Salty", "Bitter", "Pungent", "Astringent"],
    "Q31": ["Cold", "Warm", "Any", "None"],
    "Q32": ["Low", "Medium", "High", "Variable"],
    "Q33": ["Always yes", "If excess is taken causes indigestion otherwise yes", "Always with difficulty", "Cannot say"],
    "Q34": ["Butter", "Ghee", "Cheese", "Animal Fat", "Oil or oily articles", "None"],
    "Q35": ["Higher compared to others", "Lower compared to others", "Average", "Variable"],
    "Q36": ["Profuse", "Moderate", "Less", "Variable"],
    "Q37": ["Less sleep (<6 hrs)", "Moderate sleep (6-8hrs)", "Heavy sleep (>8hrs)", "Variable"],
    "Q38": ["Yes", "After few minutes / doing reading etc.", "No it takes long time to fall asleep"],
    "Q39": ["Deep", "Moderate/Sound", "Shallow"],
    "Q40": ["Regular", "Irregular", "Occasionally Irregular"],
    "Q41": ["Constipation", "Loose motions", "None"],
    "Q42": ["Hard", "Loose", "Soft", "Semisolid", "Medium"],
    "Q43": ["Gain weight easily and loose easily", "Difficulty in gaining weight", "Gain weight easily but loose with difficulty", "Stable"],
    "Q44": ["Strong", "Mild", "Very Mild"],
    "Q45": ["Cold", "Warm", "Both Cold & Warm", "Seasonal Transition", "All", "None"],
    "Q46": ["Cold", "Warm", "Both Cold & Warm", "Seasonal Transition", "All", "None"],
    "Q47": ["Frequently", "Rarely", "Moderately"],
    "Q48": ["Yes, mostly on its own", "No, it takes long time & effort to get cured", "Moderate efforts needed like diet, rest & medicine"],
    "Q49": ["Excessive", "Less", "Moderate"],
    "Q50": ["Low", "Feeble", "Weak", "Broken", "Rough", "Deep", "Good toned", "Sharp", "Clear", "High pitched", "Loud", "Soft, Pleasing"],
    "Q51": ["Slow", "Quick", "Medium", "Variably"],
    "Q52": ["Hand Movement", "Leg Movement", "Eyebrow Movement", "Shoulder Movement", "Overall Movement"],
    "Q53": ["Get stressed / disturbed frequently and can be counselled by others", 
            "Get stressed / disturbed easily and overcome it by own / with some time", 
            "Get stressed with difficulty and can overcome on own", 
            "Get stressed easily and cannot be counseled easily by others"],
    "Q54": ["On During routine work", "After doing extra work/ heavy work", "Not even after heavy work"],
    "Q55": ["Moderately", "Quickly", "Slowly", "Variably"],
    "Q56": ["Quickly", "Moderately", "Slowly", "Variably"],
    "Q57": ["Good", "Medium", "Poor", "Variable"],
    "Q58": ["Regular and routine observer", "Spontaneous and moderate routine observer", 
            "Loving to experiment with routines and change them very readily"],
    "Q59": ["Move around and interact with people and explore", "Be seated and keep confined to own work", 
            "Move around moderately and not sit for very long hours"]
}

# Define full question text
question_text = {
    "Q1": "First Name",
    "Q2": "Gender",
    "Q3": "Permanent Address",
    "Q4": "Phone Number",
    "Q5": "Email Address",
    "Q6": "Code No",
    "Q7": "Age",
    "Q8": "Date",
    "Q9": "Interviewer's Name",
    "Q10": "Are you willing to participate in the genetic analysis in Ayurgenomics project?",
    "Q11": "What is your body weight (Kg)?",
    "Q12": "What is your height (cm)?",
    "Q13": "What is your body frame?",
    "Q14": "What is your body build (bulk)?",
    "Q15": "What is the nature of your musculature?",
    "Q16": "What is the length of your forehead?",
    "Q17": "What is the texture of your nails?",
    "Q18": "What is the color of your nails?",
    "Q19": "What is the size of your finger nails?",
    "Q20": "What is the appearance of your skin? (Select all that apply)",
    "Q21": "What is the color/complexion of your skin?",
    "Q22": "What is the nature of your skin?",
    "Q23": "What is the texture of your skin?",
    "Q24": "What is the texture of your hair?",
    "Q25": "Is your scalp hair prone to? (Select all that apply)",
    "Q26": "What is the nature of your hair?",
    "Q27": "Do you observe you have? (Select all that apply)",
    "Q28": "How is your appetite (regularity)?",
    "Q29": "How is your appetite (frequency)?",
    "Q30": "What is your taste preference? (Select all that apply)",
    "Q31": "What type of food/beverages do you prefer?",
    "Q32": "How much quantity of food can you consume on feeling hungry?",
    "Q33": "Are you able to digest the amount of food consumed by you?",
    "Q34": "Do you prefer to take food rich in fats like? (Select all that apply)",
    "Q35": "Does your body temperature in general remain?",
    "Q36": "How about your perspiration?",
    "Q37": "How about your sleep (amount)?",
    "Q38": "Do you get sleep immediately after going to bed?",
    "Q39": "What is the quality of your sleep?",
    "Q40": "How about your bowel habits?",
    "Q41": "Do you tend to have?",
    "Q42": "What is the consistency of your stool?",
    "Q43": "How about changes in your body weight?",
    "Q44": "Do you have body odor?",
    "Q45": "Which weather do you prefer?",
    "Q46": "In which weather do you have health problems?",
    "Q47": "How frequently do you fall ill?",
    "Q48": "If you fall ill, do you get cured easily?",
    "Q49": "What is the amount of your speaking?",
    "Q50": "What is the quality of your voice?",
    "Q51": "What is the speed/style of your speaking?",
    "Q52": "What is the level of your voluntary and involuntary movements?",
    "Q53": "What is your mental strength?",
    "Q54": "How frequently do you feel tired?",
    "Q55": "How quickly can you memorize things?",
    "Q56": "How forgetful are you?",
    "Q57": "What is your memory retention power?",
    "Q58": "Are you a...?",
    "Q59": "Do you like to?"
}

# Map questions to features (Q13–Q59 only)
feature_mapping = {
    "Q13": ["F3"], "Q14": ["F4"], "Q15": ["F5"], "Q16": ["F6"], 
    "Q17": ["F7"], "Q18": ["F8"], "Q19": ["F9"], 
    "Q20": ["F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18"],
    "Q21": ["F19"], "Q22": ["F20"], "Q23": ["F21"], "Q24": ["F22"], 
    "Q25": ["F23", "F24", "F25", "F26"],
    "Q26": ["F27"], 
    "Q27": ["F28", "F29", "F30", "F31"],
    "Q28": ["F32"], "Q29": ["F33"], 
    "Q30": ["F34", "F35", "F36", "F37", "F38", "F39"],
    "Q31": ["F40"], "Q32": ["F41"], "Q33": ["F42"], "Q34": ["F43"], "Q35": ["F44"], "Q36": ["F45"], 
    "Q37": ["F46"], "Q38": ["F47"], "Q39": ["F48"], "Q40": ["F49"], "Q41": ["F50"], "Q42": ["F51"], 
    "Q43": ["F52"], "Q44": ["F53"], "Q45": ["F54"], "Q46": ["F55"], "Q47": ["F56"], "Q48": ["F57"], 
    "Q49": ["F58"], "Q50": ["F59"], "Q51": ["F60"], 
    "Q52": ["F61", "F62", "F63", "F64", "F65"],
    "Q53": ["F66"], "Q54": ["F67"], "Q55": ["F68"], "Q56": ["F69"], "Q57": ["F70"], "Q58": ["F71"], 
    "Q59": ["F72"]
}

# Additional features (F73 to F132)
additional_features = {
    "Q20": {
        "Cracked": "F73", "Lustrous": "F74", "Rough": "F75", "Moles": "F76", "Marks": "F77", 
        "Pimples": "F78", "Freckles": "F79", "Wrinkles": "F80"
    },
    "Q25": {
        "Graying": "F81", "Falling": "F82", "Breaking": "F83", "Split at ends": "F84"
    },
    "Q27": {
        "Nails Brittle": "F85", "Palm Cracked": "F86", "Sole Cracked": "F87", "Lips Cracked": "F88"
    },
    "Q30": {
        "Sweet": ["F89", "F90"],
        "Sour": ["F91", "F92"],
        "Salty": ["F93", "F94"],
        "Bitter": ["F95", "F96"],
        "Pungent": ["F97", "F98"],
        "Astringent": ["F99", "F100"]
    },
    "Q34": {
        "Butter": ["F101", "F102"],
        "Ghee": ["F103", "F104"],
        "Cheese": ["F105", "F106"],
        "Animal Fat": ["F107", "F108"],
        "Oil or oily articles": ["F109", "F110"]
    },
    "Q52": {
        "Hand Movement": ["F111", "F112", "F113"],
        "Leg Movement": ["F114", "F115", "F116"],
        "Eyebrow Movement": ["F117", "F118", "F119"],
        "Shoulder Movement": ["F120", "F121", "F122"],
        "Overall Movement": ["F123", "F124", "F125"]
    },
    "Others": {
        "Non_Argumentative": "F126", "AvoidConfrontations": "F127", "Convincing": "F128",
        "Non_Deviatedfrommaintopic": "F129", "Non_Irrelevantinbetween": "F130",
        "Non_Brittle/Cracked": "F131", "Non_Loose": "F132"
    }
}

# Streamlit app
st.title("Prakriti Assessment Tool")
st.write("Please provide your personal information and answer the Prakriti screening questions to determine your Vata, Pitta, and Kapha dosha percentages.")

# Collect user responses
responses = {}
with st.form("prakriti_form"):
    st.subheader("Personal Information")
    for q in [f"Q{i}" for i in range(1, 13)]:  # Q1–Q12
        question_label = question_text.get(q, q)
        if question_options[q] is None:
            responses[q] = st.text_input(question_label, key=q)
        else:
            responses[q] = st.selectbox(question_label, question_options[q], key=q)

    st.subheader("Prakriti Screening Questions")
    for q in [f"Q{i}" for i in range(13, 60)]:
        question_label = question_text.get(q, q)
        if q in ["Q20", "Q25", "Q27", "Q30", "Q34"]:
            responses[q] = st.multiselect(question_label, question_options[q], help="Select all that apply.", key=q)
        elif q == "Q52":
            for sub_q in question_options[q]:
                sub_label = f"{sub_q} level (related to {question_label})"
                responses[f"{q}_{sub_q}"] = st.selectbox(sub_label, ["High/Excessive", "Less", "Moderate"], key=f"{q}_{sub_q}")
        else:
            responses[q] = st.selectbox(question_label, question_options[q], key=q)
    
    submitted = st.form_submit_button("Submit")

# Process submission
if submitted:
    # Initialize feature vector (Q13–Q59 only)
    feature_vector = {f"F{i}": "0" for i in range(3, 133)}  # F3–F132
    
    # For storage, include all responses
    storage_vector = responses.copy()
    storage_vector["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Map responses to features (Q13–Q59)
    for q, response in responses.items():
        if q in feature_mapping and not q.startswith("Q52_"):
            if q in ["Q20", "Q25", "Q27", "Q30", "Q34"]:
                for opt in question_options[q]:
                    if q == "Q30":
                        like_feature = additional_features[q][opt][0]
                        donotlike_feature = additional_features[q][opt][1]
                        feature_vector[like_feature] = "like_" + opt if opt in response else "donotlike_" + opt
                    elif q == "Q34" and opt in additional_features[q]:
                        suit_feature = additional_features[q][opt][0]
                        donotsuit_feature = additional_features[q][opt][1]
                        feature_vector[suit_feature] = "suit_" + opt.split()[0] if opt in response else "donotsuit_" + opt.split()[0]
                    elif opt in additional_features.get(q, {}):
                        feature = additional_features[q][opt]
                        feature_vector[feature] = opt if opt in response else "Non_" + opt
            else:
                for feature in feature_mapping[q]:
                    feature_vector[feature] = response
        elif q.startswith("Q52_"):
            sub_q = q.split("_")[1]
            feature_base = additional_features["Q52"][sub_q]
            if response == "High/Excessive":
                feature_vector[feature_base[0]] = response
            elif response == "Less":
                feature_vector[feature_base[1]] = response
            else:
                feature_vector[feature_base[2]] = response
    
    # Convert feature vector to DataFrame for prediction
    input_df = pd.DataFrame([feature_vector])
    
    # Encode features
    for column in input_df.columns:
        if column in label_encoders:
            try:
                input_df[column] = label_encoders[column].transform(input_df[column].astype(str))
            except ValueError:
                input_df[column] = label_encoders[column].transform([label_encoders[column].classes_[0]])[0]
    
    # Predict probabilities
    probs = model.predict_proba(input_df)[0]
    dosha_names = target_encoder.inverse_transform([0, 1, 2])
    percentages = {dosha: prob * 100 for dosha, prob in zip(dosha_names, probs)}
    
    # Store responses
    response_df = pd.DataFrame([storage_vector])
    csv_file = "user_responses.csv"
    if os.path.exists(csv_file):
        response_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        response_df.to_csv(csv_file, mode='w', header=True, index=False)
    
    # Display results
    st.subheader("Your Prakriti Assessment Results")
    for dosha, percentage in percentages.items():
        st.write(f"{dosha}: {percentage:.2f}%")
    
    # Pie chart
    fig, ax = plt.subplots()
    ax.pie(percentages.values(), labels=percentages.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)