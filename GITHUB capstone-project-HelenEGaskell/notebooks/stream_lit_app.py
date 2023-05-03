{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "71a4b65b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import streamlit as st\n",
    "import turicreate as tc\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2d20e0be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>customer_id</th>\n",
       "      <th>article_ids</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>[610776002 706016001 706016002 782760013 74917...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0024dea548c64fb75a563e0b300c0b16210decee446f1a...</td>\n",
       "      <td>[706016001 562245001 841992002 706388002 51332...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>00357b192b81fc83261a45be87f5f3d59112db7d117513...</td>\n",
       "      <td>[610776002 706016001 751674001 905492002 90338...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0036a44bd648ce2dbc32688a465b9628b7a78395302f26...</td>\n",
       "      <td>[448509014 706016002 759871002 706016003 72012...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0040e2fc2d1e7931a38355aca56b2c62b87e65051b7287...</td>\n",
       "      <td>[706016001 610776002 759871002 706016002 44850...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11061</th>\n",
       "      <td>11061</td>\n",
       "      <td>ffddc52a24cd9e170570b48773779eec8ad05bd0cf8163...</td>\n",
       "      <td>[706016001 610776001 706016002 706016003 44850...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11062</th>\n",
       "      <td>11062</td>\n",
       "      <td>ffe6376eb6b854d842e5a7714ea758de127f086a60d67d...</td>\n",
       "      <td>[706016001 706016002 871005003 448509014 68853...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11063</th>\n",
       "      <td>11063</td>\n",
       "      <td>fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e5...</td>\n",
       "      <td>[706016001 706016002 448509014 610776002 72012...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11064</th>\n",
       "      <td>11064</td>\n",
       "      <td>fffae8eb3a282d8c43c77dd2ca0621703b71e90904dfde...</td>\n",
       "      <td>[706016002 610776002 706016003 562245001 70601...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11065</th>\n",
       "      <td>11065</td>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>[706016001 610776002 706016002 610776001 56224...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>11066 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Unnamed: 0                                        customer_id  \\\n",
       "0               0  000fb6e772c5d0023892065e659963da90b1866035558e...   \n",
       "1               1  0024dea548c64fb75a563e0b300c0b16210decee446f1a...   \n",
       "2               2  00357b192b81fc83261a45be87f5f3d59112db7d117513...   \n",
       "3               3  0036a44bd648ce2dbc32688a465b9628b7a78395302f26...   \n",
       "4               4  0040e2fc2d1e7931a38355aca56b2c62b87e65051b7287...   \n",
       "...           ...                                                ...   \n",
       "11061       11061  ffddc52a24cd9e170570b48773779eec8ad05bd0cf8163...   \n",
       "11062       11062  ffe6376eb6b854d842e5a7714ea758de127f086a60d67d...   \n",
       "11063       11063  fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e5...   \n",
       "11064       11064  fffae8eb3a282d8c43c77dd2ca0621703b71e90904dfde...   \n",
       "11065       11065  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   \n",
       "\n",
       "                                             article_ids  \n",
       "0      [610776002 706016001 706016002 782760013 74917...  \n",
       "1      [706016001 562245001 841992002 706388002 51332...  \n",
       "2      [610776002 706016001 751674001 905492002 90338...  \n",
       "3      [448509014 706016002 759871002 706016003 72012...  \n",
       "4      [706016001 610776002 759871002 706016002 44850...  \n",
       "...                                                  ...  \n",
       "11061  [706016001 610776001 706016002 706016003 44850...  \n",
       "11062  [706016001 706016002 871005003 448509014 68853...  \n",
       "11063  [706016001 706016002 448509014 610776002 72012...  \n",
       "11064  [706016002 610776002 706016003 562245001 70601...  \n",
       "11065  [706016001 610776002 706016002 610776001 56224...  \n",
       "\n",
       "[11066 rows x 3 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('dataframe.csv')\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9f415bce",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'customer_id' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[7], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# def get_article_ids(grouped, customer_id):\u001b[39;00m\n\u001b[0;32m----> 2\u001b[0m article_ids \u001b[38;5;241m=\u001b[39m df[df[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcustomer_id\u001b[39m\u001b[38;5;124m'\u001b[39m] \u001b[38;5;241m==\u001b[39m \u001b[43mcustomer_id\u001b[49m][\u001b[38;5;124m'\u001b[39m\u001b[38;5;124marticle_id\u001b[39m\u001b[38;5;124m'\u001b[39m]\u001b[38;5;241m.\u001b[39mtolist()\n\u001b[1;32m      3\u001b[0m \u001b[38;5;66;03m#     return article_ids\u001b[39;00m\n\u001b[1;32m      4\u001b[0m article_ids\n",
      "\u001b[0;31mNameError\u001b[0m: name 'customer_id' is not defined"
     ]
    }
   ],
   "source": [
    "# def get_article_ids(grouped, customer_id):\n",
    "article_ids = df[df['customer_id'] == customer_id]['article_id'].tolist()\n",
    "#     return article_ids\n",
    "article_ids"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5c8e6ee4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-04-12 17:22:21.074 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Users/helengaskell/opt/anaconda3/envs/bigdata_ml/lib/python3.8/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Define the Streamlit app\n",
    "def app():\n",
    "    # Add a title to the app\n",
    "    st.title('Get Article IDs for Customer')\n",
    "\n",
    "    # Add a text input for the customer ID\n",
    "    customer_id = st.text_input('Enter the customer ID:', '')\n",
    "\n",
    "    # If the user has entered a customer ID\n",
    "    if customer_id:\n",
    "        # Get the article IDs for the customer\n",
    "        article_ids = get_article_ids(grouped, customer_id)\n",
    "\n",
    "        # Display the article IDs\n",
    "        if article_ids:\n",
    "            st.write(f\"Article IDs for customer {customer_id}: {article_ids}\")\n",
    "        else:\n",
    "            st.write(f\"No article IDs found for customer {customer_id}.\")\n",
    "\n",
    "# Run the app\n",
    "if __name__ == '__main__':\n",
    "    app()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54c20e45",
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
