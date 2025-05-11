
# streamlit run C:\Users\admin\Desktop\Python\Learn\Intern-1\app.py for running file and to get a local site for it.


import streamlit as st
import scipy.stats as stats

st.title('Hypothesis Test')
st.text('Please give necessary input required')

def perform_ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level):
    # Convert input to integers
    control_visitors = int(control_visitors)
    control_conversions = int(control_conversions)
    treatment_visitors = int(treatment_visitors)
    treatment_conversions = int(treatment_conversions)
    
    # Calculate conversion rates for control and treatment groups
    control_rate = control_conversions / control_visitors
    treatment_rate = treatment_conversions / treatment_visitors
    
    # Calculate z-score
    z_score = (treatment_rate - control_rate) / ((control_rate * (1 - control_rate) / control_visitors) + (treatment_rate * (1 - treatment_rate) / treatment_visitors)) ** 0.5
    
    # Determine critical z-value based on confidence level
    if confidence_level == '90':
        critical_z = stats.norm.ppf(0.90)
    elif confidence_level == '95':
        critical_z = stats.norm.ppf(0.95)
    elif confidence_level == '99':
        critical_z = stats.norm.ppf(0.99)
    else:
        raise ValueError("Confidence level must be '90', '95', or '99'")
    
    # Compare z-score with critical z-value
    if z_score > critical_z:
        return "Treatment Group is Better"
    elif z_score < -critical_z:
        return "Control Group is Better"
    else:
        return "Indeterminate"

def main():
    control_visitors = st.text_input("Control Group Visitors")
    control_conversions = st.text_input("Control Group Conversions")
    treatment_visitors = st.text_input("Treatment Group Visitors")
    treatment_conversions = st.text_input("Treatment Group Conversions")
    confidence_level = st.selectbox("Confidence Level", ['90', '95', '99'])

    if st.button("Run Hypothesis Test"):
        result = perform_ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level)
        st.write(f"AB Test Result: {result}")

if __name__ == "__main__":
    main()
