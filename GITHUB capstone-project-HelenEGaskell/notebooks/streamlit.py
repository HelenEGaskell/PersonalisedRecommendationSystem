{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "23f55777",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-04-13 09:08:49.766 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Users/helengaskell/opt/anaconda3/envs/bigdata_ml/lib/python3.8/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import streamlit as st\n",
    "import turicreate as tc\n",
    "import joblib\n",
    "\n",
    "df = pd.read_csv('dataframe.csv')\n",
    "\n",
    "# Set page title\n",
    "st.title('Customer and Article IDs')\n",
    "\n",
    "# Display data table\n",
    "st.write(df)\n",
    "\n",
    "# Add a text input for the customer ID\n",
    "customer_id = st.text_input('Enter customer ID')\n",
    "\n",
    "# Get the corresponding article ID for the entered customer ID\n",
    "if customer_id:\n",
    "    try:\n",
    "        article_id = df[df['customer_id'] == int(customer_id)]['article_id'].iloc[0]\n",
    "        st.write('Article ID:', article_id)\n",
    "    except IndexError:\n",
    "        st.write('No article found for the entered customer ID')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "624afd7e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf1fc2c1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "bigdata_ml",
   "language": "python",
   "name": "bigdata_ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
