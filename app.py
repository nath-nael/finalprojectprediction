import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Salary Cluster Predictor",
    page_icon="💰",
    layout="wide"
)

# Load all saved models and encoders
@st.cache_resource
def load_artifacts():
    """Load all the necessary ML artifacts"""
    try:
        return {
            'model': joblib.load("ridge_model.pkl"),
            'scaler': joblib.load("scaler.pkl"),
            'target_encoders': {
                'Country_cleaned': joblib.load("target_encoder_Country_cleaned.pkl"),
                'Industry_grouped': joblib.load("target_encoder_Industry_grouped.pkl"),
                'JobTitle_grouped': joblib.load("target_encoder_JobTitle_grouped.pkl")
            },
            'gender_mapping': joblib.load("gender_mapping.pkl"),
            'salary_map': joblib.load("cluster_salary_map.pkl")
        }
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None

# Load artifacts
artifacts = load_artifacts()

if artifacts is None:
    st.stop()

# Title and description
st.title("💰 Salary Cluster Prediction")
st.markdown("""
This app predicts which salary cluster an employee may belong to, based on their professional and demographic profile.
""")

# Create input form
with st.form("prediction_form"):
    st.header("Employee Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Details")
        gender = st.selectbox("Gender", ['Man', 'Woman', 'Non-binary', 'Other or prefer not to answer'])
        country = st.selectbox("Country", ['Belgium', 'Poland', 'United Kingdom', 'Canada', 'Italy',
       'Netherlands', 'United States', 'France', 'Ireland', 'Australia',
       'Germany', 'Singapore', 'Denmark', 'South Africa', 'Finland',
       'Nigeria', 'Colombia', 'Sweden', 'Japan', 'India', 'Eritrea',
       'Bermuda', 'New Zealand', 'China', 'Spain', 'Malaysia', 'Kenya',
       'Croatia', 'Austria', 'Greece', 'Pakistan', 'Brazil', 'HongKong',
       'Norway', 'Philippines', 'Israel', 'Portugal', 'Hong Kong',
       'Indonesia', 'Egypt', 'Uruguay', 'Mexico', 'Thailand',
       'Sierra Leone', 'Switzerland', 'Sri Lanka', 'Morocco', 'Argentina',
       'Bosnia and Herzegovina', 'Hungary', 'Slovakia', 'Bangladesh',
       'Zimbabwe', 'Chile', 'Puerto Rico', 'Lithuania', 'Slovenia',
       'Jordan', 'Latvia', 'Isle of Man', 'Qatar', 'Estonia', 'Serbia',
       'Saudi Arabia', 'Cyprus', 'Romania', 'Myanmar', 'Ghana',
       'Cayman Islands', 'Luxembourg', 'Czechia', 'Costa Rica',
       'Trinidad and Tobago', 'Malta', 'Uganda', 'Cuba', 'Ukraine',
       'Kuwait', "Côte d'Ivoire", 'Bulgaria', 'Rwanda', 'Somalia'])
    
    with col2:
        st.subheader("Professional Details")
        industry = st.selectbox("Industry", [ 'Tech', 'Consulting', 'Engineering & Manufacturing', 'Healthcare',
    'Other', 'Legal', 'Marketing', 'Finance', 'Government',
    'Education', 'Construction & Real Estate', 'Retail', 'Agriculture',
    'Logistics', 'Art & Design', 'Media & Entertainment', 'Nonprofit','Other'])
        job_title = st.selectbox("Job Title", [
            'Management', 'Executive', 'Engineering & IT', 'Consulting',
    'Legal', 'Administrative', 'Project & Program Management',
    'Strategist', 'Creative Arts', 'Other',
    'Marketing & Communications', 'Data & Analytics',
    'Education & Training', 'Finance & Accounting', 'Healthcare',
    'News & Media', 'Human Resources', 'Sales',
    'Researcher & Scientist', 'Customer Service', 'Assistant',
    'Intern', 'Language Expert', 'Entrepreneur', 'Other'
        ])
    
    submitted = st.form_submit_button("Predict Salary Cluster")

# When form is submitted
if submitted:
    try:
        # Create a dataframe from inputs with exact column order as training
        # Common orders: Gender first, then categorical features
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Country_cleaned': [country],
            'Industry_grouped': [industry],
            'JobTitle_grouped': [job_title]
        })
        
        # Debug: Show original input
        # st.write("Debug - Original input:", input_data)

        # Apply target encoding FIRST, before any other transformations
        input_encoded = input_data.copy()
        
        for col in ['Country_cleaned', 'Industry_grouped', 'JobTitle_grouped']:
            if col in artifacts['target_encoders']:
                try:
                    # Handle unknown categories by using a default value
                    encoder = artifacts['target_encoders'][col]
                    
                    # Check if the value exists in the encoder
                    if hasattr(encoder, 'mapping_'):
                        # For category_encoders target encoder
                        known_values = encoder.mapping_[0]['mapping'].keys()
                        if input_data[col].iloc[0] not in known_values:
                            # Use the mean of all encoded values as default
                            default_value = np.mean(list(encoder.mapping_[0]['mapping'].values()))
                            input_encoded[col] = [default_value]
                        else:
                            input_encoded[col] = encoder.transform(input_data[col])
                    else:
                        # For sklearn-style encoders
                        input_encoded[col] = encoder.transform(input_data[col])
                        
                except Exception as e:
                    st.warning(f"Issue with encoding {col}: {e}")
                    # Use a default encoded value (e.g., 0 or mean)
                    input_encoded[col] = [0.0]
        
        # Apply gender mapping
        if 'gender_mapping' in artifacts:
            input_encoded['Gender'] = input_data['Gender'].map(artifacts['gender_mapping'])
            # Fill any unmapped values with a default (e.g., the most common encoding)
            input_encoded['Gender'] = input_encoded['Gender'].fillna(2)
        else:
            # Create a simple gender encoding if mapping doesn't exist
            gender_map = {'Man': 0, 'Woman': 1, 'Non-binary': 2, 'Other or prefer not to answer': 3}
            input_encoded['Gender'] = input_data['Gender'].map(gender_map).fillna(2)
        
        # Debug: Show encoded input
        # st.write("Debug - After encoding:", input_encoded)
        
        # Ensure correct column order - try different common orders
        possible_orders = [
            ['Gender', 'Country_cleaned', 'Industry_grouped', 'JobTitle_grouped'],  # Original order
            ['Country_cleaned', 'Industry_grouped', 'JobTitle_grouped', 'Gender'],  # Categorical first
            ['Gender', 'JobTitle_grouped', 'Industry_grouped', 'Country_cleaned'],  # Alternative order
        ]
        
        prediction_successful = False
        
        for col_order in possible_orders:
            try:
                # Reorder columns
                input_ordered = input_encoded[col_order]
                
                # Debug: Show column order being tried
                # st.write(f"Debug - Trying column order: {col_order}")
                # st.write("Debug - Data shape:", input_ordered.shape)
                # st.write("Debug - Data types:", input_ordered.dtypes)
                
                # Scale the data
                scaled_data = artifacts['scaler'].transform(input_ordered)
                
                # Debug: Show scaled data
                # st.write("Debug - Scaled data shape:", scaled_data.shape)
                
                # Predict cluster
                cluster_proba = artifacts['model'].predict(scaled_data)
                cluster = int(cluster_proba[0]) if hasattr(cluster_proba, '__len__') else int(cluster_proba)
                
                prediction_successful = True
                break
                
            except Exception as e:
                # st.write(f"Debug - Failed with order {col_order}: {str(e)}")
                continue
        
        if not prediction_successful:
            raise Exception("Could not find correct feature order. Please check your model training data.")
        
        # Get average salary
        salary_map = artifacts['salary_map']
        avg_salary = salary_map.get(cluster, None)

        # Output prediction
        st.success(f"🎯 Predicted Salary Cluster: **{cluster}**")
        if avg_salary:
            st.markdown(f"💵 **Estimated Average Salary** for this cluster: **${avg_salary:,.2f}** a year")
        else:
            st.warning("No salary data found for this cluster.")

        # Cluster interpretation
        st.subheader("Cluster Description")
        cluster_descriptions = {
            0: "Entry-level positions with lower compensation",
            1: "Early-career professionals with moderate compensation",
            2: "Mid-level professionals with competitive salaries",
            3: "Experienced professionals with above-average compensation",
            4: "Senior roles with very high compensation",
            5: "Executive/leadership positions with top-tier compensation",
            6: "Specialized high-paying experts or startup founders",
            7: "High-value tech/business entrepreneurs",
            8: "Niche leaders in AI/Finance sectors",
            9: "Internationally mobile top executives"
        }
        st.info(cluster_descriptions.get(cluster, "No description available for this cluster."))

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        
        # Additional debugging information
        st.subheader("Debug Information")
        st.write("If you're still getting errors, please check:")
        st.write("1. That all .pkl files are in the same directory as this script")
        st.write("2. The feature names and order match your training data")
        st.write("3. The target encoders were saved properly")
        
        # Try to show model feature names if available
        try:
            if hasattr(artifacts['model'], 'feature_names_in_'):
                st.write("Model expects features:", artifacts['model'].feature_names_in_)
            elif hasattr(artifacts['scaler'], 'feature_names_in_'):
                st.write("Scaler expects features:", artifacts['scaler'].feature_names_in_)
        except:
            st.write("Could not retrieve expected feature names from model")

# Sidebar Info
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
This app uses machine learning to classify employee profiles into salary clusters.
- **Model**: Ridge Classifier
- **Encoding**: Target Encoding for Categorical Data
- **Scaling**: Standard Scaler for features
- **Prediction**: Returns the cluster and expected average salary

---
🔒 *Note: This is a prototype tool. Results are based on training data trends.*
    """)
    
    # Show loaded artifacts status
    st.header("🔧 System Status")
    if artifacts:
        st.success("✅ All models loaded successfully")
        
        # Show feature info if available
        try:
            if hasattr(artifacts['scaler'], 'feature_names_in_'):
                st.write("**Expected Features:**")
                for i, feature in enumerate(artifacts['scaler'].feature_names_in_):
                    st.write(f"{i+1}. {feature}")
        except:
            pass
    else:
        st.error("❌ Failed to load models")