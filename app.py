import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load('prakriti_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Define questionnaire options (based on Prakriti Questions.pdf)
question_options = {
    'Q13': ['Narrow', 'Medium', 'Wide'],
    'Q14': ['Weakly developed', 'Moderately developed', 'Well developed'],
    'Q15': ['Thin Musculature', 'Soft and Loosely knitted Musculature', 'Smooth and Firmly knitted Musculature'],
    'Q16': ['Small', 'Medium', 'Large'],
    'Q17': ['Smooth', 'Soft', 'Rough'],
    'Q18': ['Reddish', 'Pale', 'Pink'],
    'Q19': ['Small', 'Medium', 'Large'],
    'Q20': ['Cracked', 'Lustrous', 'Rough', 'Moles', 'Marks', 'Pimples', 'Freckles', 'Wrinkles', 'None'],
    'Q21': ['Fair with reddish tinge', 'Fair with yellowish tinge', 'Dark', 'Dusky', 'Wheatish', 'Fair with pale tinge', 'Fair with pink tinge'],
    'Q22': ['Dry', 'Oily', 'Normal', 'Seasonal'],
    'Q23': ['Thick', 'Thin'],
    'Q24': ['Thick', 'Thin'],
    'Q25': ['Graying', 'Falling', 'Breaking', 'Split at ends', 'Both', 'None'],
    'Q26': ['Dry', 'Oily', 'Normal', 'Seasonal'],
    'Q27_Nails': ['Brittle', 'Non_Brittle'],
    'Q27_Palm': ['Cracked', 'Non_Cracked'],
    'Q27_Sole': ['Cracked', 'Non_Cracked'],
    'Q27_Lips': ['Cracked', 'Non_Cracked'],
    'Q28': ['Regular', 'Irregular'],
    'Q29': ['Frequent', 'Infrequent'],
    'Q30_Sweet': ['LikeSweet', 'DonotlikeSweet'],
    'Q30_Sour': ['LikeSour', 'DonotLikeSour'],
    'Q30_Salty': ['LikeSalty', 'DoNotLikeSalty'],
    'Q30_Bitter': ['LikeBitter', 'DonotlikeBitter'],
    'Q30_Pungent': ['LikePungent', 'Donotlike_Pungent'],
    'Q30_Astringent': ['Like_Astringent', 'Donotlike_Astringent'],
    'Q31': ['Cold', 'Warm', 'Any', 'None'],
    'Q32': ['Low', 'Medium', 'High', 'Variable'],
    'Q33': ['Always yes', 'If excess is taken causes indigestion otherwise yes', 'Always with difficulty', 'Cannot say'],
    'Q34': ['Butter', 'Ghee', 'Cheese', 'Animal Fat', 'Oil or oily articles', 'None'],
    'Q35': ['Higher compared to others', 'Lower compared to others', 'Average', 'Variable'],
    'Q36': ['Profuse', 'Moderate', 'Less', 'Variable'],
    'Q37': ['Less sleep (<6 hrs)', 'Moderate sleep (6-8hrs)', 'Heavy sleep (>8hrs)', 'Variable'],
    'Q38': ['Yes', 'After few minutes / doing reading etc.', 'No it takes long time to fall asleep'],
    'Q39': ['Deep', 'Moderate/Sound', 'Shallow'],
    'Q40': ['Regular', 'Irregular', 'Occasionally Irregular'],
    'Q41': ['Constipation', 'Loose motions', 'None'],
    'Q42': ['Hard', 'Loose', 'Soft', 'Semisolid', 'Medium'],
    'Q43': ['Gain weight easily and lose easily', 'Difficulty in gaining weight', 'Gain weight easily but lose with difficulty', 'Stable'],
    'Q44': ['Strong', 'Mild', 'Very Mild'],
    'Q45': ['Cold', 'Warm', 'Both Cold & Warm', 'Seasonal Transition', 'All', 'None'],
    'Q46': ['Cold', 'Warm', 'Both Cold & Warm', 'Seasonal Transition', 'All', 'None'],
    'Q47': ['Frequently', 'Rarely', 'Moderately'],
    'Q48': ['Yes, mostly on its own', 'No, it takes long time & effort to get cured', 'Moderate efforts needed like diet, rest & medicine'],
    'Q49': ['Excessive', 'Less', 'Moderate'],
    'Q50': ['Low', 'Feeble', 'Weak', 'Broken', 'Rough', 'Deep', 'Good toned', 'Sharp', 'Clear', 'High pitched', 'Loud', 'Soft, Pleasing'],
    'Q51': ['Slow', 'Quick', 'Medium', 'Variably'],
    'Q52_Hand': ['High/Excessive', 'Less', 'Moderate'],
    'Q52_Leg': ['High/Excessive', 'Less', 'Moderate'],
    'Q52_Eyebrow': ['High/Excessive', 'Less', 'Moderate'],
    'Q52_Shoulder': ['High/Excessive', 'Less', 'Moderate'],
    'Q52_Overall': ['High/Excessive', 'Less', 'Moderate'],
    'Q53': ['Get stressed / disturbed frequently and can be counselled by others',
             'Get stressed / disturbed easily and overcome it by own / with some time',
             'Get stressed with difficulty and can overcome on own',
             'Get stressed easily and cannot be counseled easily by others'],
    'Q54': ['On During routine work', 'After doing extra work/ heavy work', 'Not even after heavy work'],
    'Q55': ['Moderately', 'Quickly', 'Slowly', 'Variably'],
    'Q56': ['Quickly', 'Moderately', 'Slowly', 'Variably'],
    'Q57': ['Good', 'Medium', 'Poor', 'Variable'],
    'Q58': ['Regular and routine observer', 'Spontaneous and moderate routine observer',
            'Loving to experiment with routines and change them very readily'],
    'Q59': ['Move around and interact with people and explore',
            'Be seated and keep confined to own work',
            'Move around moderately and not sit for very long hours']
}

# Streamlit app
st.title("Prakriti Dosha Assessment")
st.write("Answer the following questions to determine your Vata, Pitta, and Kapha percentages.")

# Collect user inputs
user_inputs = {}
for q, options in question_options.items():
    user_inputs[q] = st.selectbox(q.replace('_', ' '), options)

# Process inputs
if st.button("Calculate Dosha Percentages"):
    # Map user inputs to features (F1 to F132)
    feature_values = []
    feature_columns = [f'F{i}' for i in range(1, 133)]  # F1 to F132
    feature_idx = 0
    for q, response in user_inputs.items():
        if q.startswith('Q27_') or q.startswith('Q30_') or q.startswith('Q52_'):
            # Handle multi-option questions
            feature_values.append(response)
            feature_idx += 1
        else:
            feature_values.append(response)
            feature_idx += 1
            # Add additional features for multi-option questions if needed
            if q == 'Q20':  # Skin Appearance (multiple binary options)
                for option in ['Cracked', 'Lustrous', 'Rough', 'Moles', 'Marks', 'Pimples', 'Freckles', 'Wrinkles']:
                    feature_values.append(option if response == option else f'Non_{option}')
                    feature_idx += 1
            elif q == 'Q25':  # Hair Nature (multiple binary options)
                for option in ['Graying', 'Falling', 'Breaking', 'Split at ends']:
                    feature_values.append(option if response == option else f'Non_{option}')
                    feature_idx += 1

    # Create DataFrame for prediction
    input_df = pd.DataFrame([feature_values[:132]], columns=feature_columns)

    # Encode inputs
    for column in input_df.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                input_df[column] = le.transform(input_df[column].astype(str))
            except ValueError:
                input_df[column] = 0  # Default value for unseen categories

    # Predict
    probs = model.predict_proba(input_df)[0]
    dosha_names = target_encoder.inverse_transform([0, 1, 2])  # Adjust based on actual encoding
    percentages = probs * 100

    # Display results
    st.subheader("Your Dosha Percentages")
    for dosha, perc in zip(dosha_names, percentages):
        st.write(f"{dosha}: {perc:.2f}%")

    # Plot pie chart
    fig, ax = plt.subplots()
    ax.pie(percentages, labels=dosha_names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# Instructions
st.write("Note: Ensure all questions are answered accurately for the best results.")
