import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataframe.csv', index_col=0)

# Set page title
st.title('H&M Personalised Recommendations')

# Display data table
st.write(df)

# Add a text input for the customer ID
customer_id = st.text_input('Enter customer ID')
customer_index = df.loc[df['customer_id'] == customer_id].index[0]
customer_index
recommendations = df.iloc[customer_index, 1]
st.write(recommendations)


image = Image.open('0430694018.jpg')
image1 = Image.open('0461414008.jpg')
image2 = Image.open('0540241002.jpg')
image3 = Image.open('0710056002.jpg')
image4 = Image.open('0730057001.jpg')
image5 = Image.open('0770030002.jpg')
image6 = Image.open('0770030003.jpg')
image7 = Image.open('0800016002.jpg')

st.image(image)
st.image(image1)
st.image(image2)
st.image(image3)
st.image(image4)
st.image(image5)
st.image(image6)
st.image(image7)

