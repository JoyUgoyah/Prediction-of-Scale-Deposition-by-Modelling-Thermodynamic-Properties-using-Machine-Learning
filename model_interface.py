import pickle
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time
from PIL import Image



# Insert project title as header
st.title('PREDICTION OF SCALE FORMATION BY MODELLING THERMODYNAMIC PROPERTIES USING MACHINE LEARNING')
st.markdown('Joy Ugoyah')


# Data input description
st.subheader('Input your data in the format described below')
st.markdown('Please upload an excel file of `.csv` format containing the following parameters in the correct oilfield units as indicated to avoid prediction error.')
st.markdown('Well No')
st.markdown('Temperature (oF)')
st.markdown('Pressure (psia)')
st.markdown('CO2 mole frac.')
st.markdown('pH')
st.markdown('Ca2+ (ppm)')
st.markdown('Na+ (ppm)')
st.markdown('Mg2+ (ppm)')
st.markdown('Fe2+ (ppm)')
st.markdown('HCO3- (ppm)')
st.markdown('SO4 2-(ppm)')
st.markdown('Cl- (ppm)')
st.markdown('CO3 2- (ppm)')
st.markdown('Ba2+ (ppm)')
st.markdown('TDS (ppm)')
st.markdown('Inspection Result')
st.markdown('Scale Type')


# Upload file button widget
st.subheader('Please upload your data here')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    scale_data = pd.read_csv(uploaded_file)

    

    def preprocess(scale_data=scale_data):
        
        """
        Preprocess raw input data.

        Args:
        ---
        input_data - Batch of samples. Numpy array of shape (m, 17) where m is
        the number of samples and 17 is the dimension of the feature vector.
        """

        # Rename columns for easier manipulation
        scale_data.rename(columns={'Temperature (oF)': 'T', 'Pressure (psia)':'P', 'CO2 mole frac.':'XCO2',
       'Ca2+ (ppm)':'Ca2+', 'Na+ (ppm)':'Na+', 'Mg2+ (ppm)':'Mg2+', 'Fe2+ (ppm)':'Fe2+',
       'HCO3- (ppm)':'HCO3-', 'SO4 2-(ppm)':'SO4 2-', 'Cl- (ppm)':'Cl-', 'CO3 2- (ppm)':'CO3 2-', 'Ba2+ (ppm)':'Ba2+',
       'TDS (ppm)':'TDS'}, inplace=True) 

        # Drop columns that do not appear to be of great importance of any
        scale_data.drop('Ba2+', axis=1, inplace=True)

        # Missing values imputation
        scale_data = pd.DataFrame(scale_data.fillna(scale_data.mean()))
        

    
        return scale_data
       

    # Load a fitted model from the local filesystem into memory.
    filename = open('classifier.pkl', 'rb')
    model = pickle.load(filename)
     

    def predict_batch(model, batch_input_features):
        '''
        Function that predicts a batch of sampels.

        Args:
        ---
        batch_input_features: A batch of features required by the model to
        generate predictions. Numpy array of shape (m, n) where m is the 
        number of samples and n is the dimension of the feature vector.

        Returns:
        --------
        prediction: Predictions of the model. Numpy array of shape (m,).
        '''
        # Import evaluation metric

        # Make prediction
        
        y_pred = model.predict(batch_input_features)

        # Show model prediction accuracy score and dataframe containing date and prediction
        
 
        return y_pred

    # Preprocess input data   
    preprocess()  

    # Define features
    X = scale_data.drop(columns=['Well No'], axis=1) 

    # Make prediction
    y_pred = predict_batch(model=model, batch_input_features=X) 

    # Output data frame
    output = pd.DataFrame({'Well No': scale_data['Well No'], 'Condition': y_pred})

    

    # Warning Image output
    #a = output['Prediction'].iloc[-2]
    #b = output['Prediction'].iloc[-1]

    #if (b=='Abnormal occurrence' and b==a):
    #    image = Image.open('abnormal trend.png')
        
    #    st.image(image, caption='Warning: Choke is Performing Abnormally!')

    #else:
    #    image = Image.open('normal trend.png')
        
    #    st.image(image, caption='Choke is Performing Normally.')


    #Print predictions as table
    st.table(output)
    
    # Give double line spacing
    st.markdown(' ')
    st.markdown(' ')

    # Add line graph of result
    c = alt.Chart(output, title='Scale Monitoring - Prediction').mark_line().encode(
     x='Well No', y='Condition').properties(width=800, height=300)

    st.altair_chart(c, use_container_width=True)
