{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5234db84",
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
   "execution_count": 2,
   "id": "d28863fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('../src/Data/dummy_df.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d605af97",
   "metadata": {},
   "outputs": [],
   "source": [
    "# df1 = tc.SFrame(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "ebc2ac8b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method SFrame.head of Columns:\n",
       "\tUnnamed: 0\tint\n",
       "\tcustomer_id\tstr\n",
       "\tarticle_id\tint\n",
       "\tarticle_purchase_count\tint\n",
       "\tdummy\tint\n",
       "\n",
       "Rows: 2006351\n",
       "\n",
       "Data:\n",
       "+------------+-------------------------------+------------+------------------------+\n",
       "| Unnamed: 0 |          customer_id          | article_id | article_purchase_count |\n",
       "+------------+-------------------------------+------------+------------------------+\n",
       "|     0      | 000fb6e772c5d0023892065e65... | 108775044  |           2            |\n",
       "|     1      | 000fb6e772c5d0023892065e65... | 111565001  |           1            |\n",
       "|     2      | 000fb6e772c5d0023892065e65... | 111586001  |           1            |\n",
       "|     3      | 000fb6e772c5d0023892065e65... | 111593001  |           1            |\n",
       "|     4      | 000fb6e772c5d0023892065e65... | 158340001  |           3            |\n",
       "|     5      | 000fb6e772c5d0023892065e65... | 179950002  |           1            |\n",
       "|     6      | 000fb6e772c5d0023892065e65... | 200182001  |           1            |\n",
       "|     7      | 000fb6e772c5d0023892065e65... | 214844002  |           2            |\n",
       "|     8      | 000fb6e772c5d0023892065e65... | 234432001  |           1            |\n",
       "|     9      | 000fb6e772c5d0023892065e65... | 301656017  |           1            |\n",
       "+------------+-------------------------------+------------+------------------------+\n",
       "+-------+\n",
       "| dummy |\n",
       "+-------+\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "|   1   |\n",
       "+-------+\n",
       "[2006351 rows x 5 columns]\n",
       "Note: Only the head of the SFrame is printed.\n",
       "You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.>"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# df1.head"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "2a79189b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# defining the constant variables for my turicreate models \n",
    "user_id = 'customer_id'\n",
    "item_id = 'article_id'\n",
    "# customer_to_rec defines the customers that we will be using in our model\n",
    "customers_to_rec = list(df['customer_id'].unique())\n",
    "# n_rec is the number of items we will recommend to each customer\n",
    "n_rec = 8\n",
    "# number of rows we want to see in the initial output \n",
    "n_display = 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "e4746c50",
   "metadata": {},
   "outputs": [],
   "source": [
    "def model(train_data, name, user_id, item_id, target, customers_to_rec, n_rec, n_display):\n",
    "    model = tc.item_similarity_recommender.create(train_data, \n",
    "                                                    user_id=user_id, \n",
    "                                                    item_id=item_id, \n",
    "                                                    target=target, \n",
    "                                                    similarity_type='cosine')\n",
    "        \n",
    "    recom = model.recommend(users=customers_to_rec, k=n_rec)\n",
    "    recom.print_rows(n_display)\n",
    "    return recom['article_id']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "7f89c8ff",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre>Warning: Ignoring columns Unnamed: 0, article_purchase_count;</pre>"
      ],
      "text/plain": [
       "Warning: Ignoring columns Unnamed: 0, article_purchase_count;"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    To use these columns in scoring predictions, use a model that allows the use of additional features.</pre>"
      ],
      "text/plain": [
       "    To use these columns in scoring predictions, use a model that allows the use of additional features."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Preparing data set.</pre>"
      ],
      "text/plain": [
       "Preparing data set."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    Data has 2006351 observations with 11066 users and 76548 items.</pre>"
      ],
      "text/plain": [
       "    Data has 2006351 observations with 11066 users and 76548 items."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    Data prepared in: 1.14236s</pre>"
      ],
      "text/plain": [
       "    Data prepared in: 1.14236s"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Training model from provided data.</pre>"
      ],
      "text/plain": [
       "Training model from provided data."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Gathering per-item and per-user statistics.</pre>"
      ],
      "text/plain": [
       "Gathering per-item and per-user statistics."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| Elapsed Time (Item Statistics) | % Complete |</pre>"
      ],
      "text/plain": [
       "| Elapsed Time (Item Statistics) | % Complete |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 2.69ms                         | 9          |</pre>"
      ],
      "text/plain": [
       "| 2.69ms                         | 9          |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 28.172ms                       | 100        |</pre>"
      ],
      "text/plain": [
       "| 28.172ms                       | 100        |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Setting up lookup tables.</pre>"
      ],
      "text/plain": [
       "Setting up lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Processing data in 3 passes using dense lookup tables.</pre>"
      ],
      "text/plain": [
       "Processing data in 3 passes using dense lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| Elapsed Time (Constructing Lookups) | Total % Complete | Items Processed |</pre>"
      ],
      "text/plain": [
       "| Elapsed Time (Constructing Lookups) | Total % Complete | Items Processed |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 3.47s                               | 0                | 2               |</pre>"
      ],
      "text/plain": [
       "| 3.47s                               | 0                | 2               |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 15.03s                              | 33.25            | 25516           |</pre>"
      ],
      "text/plain": [
       "| 15.03s                              | 33.25            | 25516           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 24.94s                              | 66.5             | 51032           |</pre>"
      ],
      "text/plain": [
       "| 24.94s                              | 66.5             | 51032           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 29.85s                              | 100              | 76548           |</pre>"
      ],
      "text/plain": [
       "| 29.85s                              | 100              | 76548           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Finalizing lookup tables.</pre>"
      ],
      "text/plain": [
       "Finalizing lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Generating candidate set for working with new users.</pre>"
      ],
      "text/plain": [
       "Generating candidate set for working with new users."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Finished training in 29.9623s</pre>"
      ],
      "text/plain": [
       "Finished training in 29.9623s"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 1000/11066 queries. users per second: 2525.45</pre>"
      ],
      "text/plain": [
       "recommendations finished on 1000/11066 queries. users per second: 2525.45"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 2000/11066 queries. users per second: 2506.03</pre>"
      ],
      "text/plain": [
       "recommendations finished on 2000/11066 queries. users per second: 2506.03"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 3000/11066 queries. users per second: 2507.55</pre>"
      ],
      "text/plain": [
       "recommendations finished on 3000/11066 queries. users per second: 2507.55"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 4000/11066 queries. users per second: 2504.36</pre>"
      ],
      "text/plain": [
       "recommendations finished on 4000/11066 queries. users per second: 2504.36"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 5000/11066 queries. users per second: 2505.56</pre>"
      ],
      "text/plain": [
       "recommendations finished on 5000/11066 queries. users per second: 2505.56"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 6000/11066 queries. users per second: 2508.92</pre>"
      ],
      "text/plain": [
       "recommendations finished on 6000/11066 queries. users per second: 2508.92"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 7000/11066 queries. users per second: 2511.81</pre>"
      ],
      "text/plain": [
       "recommendations finished on 7000/11066 queries. users per second: 2511.81"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 8000/11066 queries. users per second: 2503.74</pre>"
      ],
      "text/plain": [
       "recommendations finished on 8000/11066 queries. users per second: 2503.74"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 9000/11066 queries. users per second: 2501.95</pre>"
      ],
      "text/plain": [
       "recommendations finished on 9000/11066 queries. users per second: 2501.95"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 10000/11066 queries. users per second: 2504.25</pre>"
      ],
      "text/plain": [
       "recommendations finished on 10000/11066 queries. users per second: 2504.25"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 11000/11066 queries. users per second: 2504.99</pre>"
      ],
      "text/plain": [
       "recommendations finished on 11000/11066 queries. users per second: 2504.99"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------------------------------+------------+----------------------+------+\n",
      "|          customer_id          | article_id |        score         | rank |\n",
      "+-------------------------------+------------+----------------------+------+\n",
      "| 000fb6e772c5d0023892065e65... | 610776002  | 0.00963328473361922  |  1   |\n",
      "| 000fb6e772c5d0023892065e65... | 706016001  | 0.008656342382784243 |  2   |\n",
      "| 000fb6e772c5d0023892065e65... | 706016002  | 0.006214228971504871 |  3   |\n",
      "| 000fb6e772c5d0023892065e65... | 782760013  | 0.005852978023481958 |  4   |\n",
      "| 000fb6e772c5d0023892065e65... | 749175001  | 0.005852978023481958 |  5   |\n",
      "| 000fb6e772c5d0023892065e65... | 746298001  | 0.005852978023481958 |  6   |\n",
      "| 000fb6e772c5d0023892065e65... | 714083008  | 0.005852978023481958 |  7   |\n",
      "| 000fb6e772c5d0023892065e65... | 680757003  | 0.005852978023481958 |  8   |\n",
      "+-------------------------------+------------+----------------------+------+\n",
      "[88528 rows x 4 columns]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "name = 'cosine'\n",
    "target = 'dummy'\n",
    "\n",
    "cosine_dummy = model(df1, name, user_id, item_id, target, customers_to_rec, n_rec, n_display)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "1ada3335",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'recom' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[43], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[43mrecom\u001b[49m)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'recom' is not defined"
     ]
    }
   ],
   "source": [
    "print(recom)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "99378bb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df2 = pd.DataFrame(cosine_dummy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "37d666df",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "ename": "AttributeError",
     "evalue": "'function' object has no attribute 'recommend'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[42], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m recommendations \u001b[38;5;241m=\u001b[39m \u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrecommend\u001b[49m(select_columns\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcustomer_id\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124marticle_id\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mscore\u001b[39m\u001b[38;5;124m'\u001b[39m])\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'function' object has no attribute 'recommend'"
     ]
    }
   ],
   "source": [
    "recommendations = model.recommend(select_columns=['customer_id', 'article_id', 'score'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "10023325",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>610776002</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>706016001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>706016002</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>782760013</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>749175001</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           0\n",
       "0  610776002\n",
       "1  706016001\n",
       "2  706016002\n",
       "3  782760013\n",
       "4  749175001"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df2.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "33b06bb4",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre>Warning: Ignoring columns Unnamed: 0, article_purchase_count;</pre>"
      ],
      "text/plain": [
       "Warning: Ignoring columns Unnamed: 0, article_purchase_count;"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    To use these columns in scoring predictions, use a model that allows the use of additional features.</pre>"
      ],
      "text/plain": [
       "    To use these columns in scoring predictions, use a model that allows the use of additional features."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Preparing data set.</pre>"
      ],
      "text/plain": [
       "Preparing data set."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    Data has 2006351 observations with 11066 users and 76548 items.</pre>"
      ],
      "text/plain": [
       "    Data has 2006351 observations with 11066 users and 76548 items."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>    Data prepared in: 0.955226s</pre>"
      ],
      "text/plain": [
       "    Data prepared in: 0.955226s"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Training model from provided data.</pre>"
      ],
      "text/plain": [
       "Training model from provided data."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Gathering per-item and per-user statistics.</pre>"
      ],
      "text/plain": [
       "Gathering per-item and per-user statistics."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| Elapsed Time (Item Statistics) | % Complete |</pre>"
      ],
      "text/plain": [
       "| Elapsed Time (Item Statistics) | % Complete |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 2.821ms                        | 9          |</pre>"
      ],
      "text/plain": [
       "| 2.821ms                        | 9          |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 28.146ms                       | 100        |</pre>"
      ],
      "text/plain": [
       "| 28.146ms                       | 100        |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+--------------------------------+------------+</pre>"
      ],
      "text/plain": [
       "+--------------------------------+------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Setting up lookup tables.</pre>"
      ],
      "text/plain": [
       "Setting up lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Processing data in 3 passes using dense lookup tables.</pre>"
      ],
      "text/plain": [
       "Processing data in 3 passes using dense lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| Elapsed Time (Constructing Lookups) | Total % Complete | Items Processed |</pre>"
      ],
      "text/plain": [
       "| Elapsed Time (Constructing Lookups) | Total % Complete | Items Processed |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 3.55s                               | 0                | 0               |</pre>"
      ],
      "text/plain": [
       "| 3.55s                               | 0                | 0               |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 17.91s                              | 33.25            | 25516           |</pre>"
      ],
      "text/plain": [
       "| 17.91s                              | 33.25            | 25516           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 26.49s                              | 66.5             | 51032           |</pre>"
      ],
      "text/plain": [
       "| 26.49s                              | 66.5             | 51032           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>| 32.07s                              | 100              | 76548           |</pre>"
      ],
      "text/plain": [
       "| 32.07s                              | 100              | 76548           |"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>+-------------------------------------+------------------+-----------------+</pre>"
      ],
      "text/plain": [
       "+-------------------------------------+------------------+-----------------+"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Finalizing lookup tables.</pre>"
      ],
      "text/plain": [
       "Finalizing lookup tables."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Generating candidate set for working with new users.</pre>"
      ],
      "text/plain": [
       "Generating candidate set for working with new users."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>Finished training in 32.1895s</pre>"
      ],
      "text/plain": [
       "Finished training in 32.1895s"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model1 = tc.item_similarity_recommender.create(df1, user_id='customer_id', item_id='article_id', target='dummy', similarity_type='cosine')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "fcec2965",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 1000/11066 queries. users per second: 2437.58</pre>"
      ],
      "text/plain": [
       "recommendations finished on 1000/11066 queries. users per second: 2437.58"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 2000/11066 queries. users per second: 2489.86</pre>"
      ],
      "text/plain": [
       "recommendations finished on 2000/11066 queries. users per second: 2489.86"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 3000/11066 queries. users per second: 2509.97</pre>"
      ],
      "text/plain": [
       "recommendations finished on 3000/11066 queries. users per second: 2509.97"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 4000/11066 queries. users per second: 2511.66</pre>"
      ],
      "text/plain": [
       "recommendations finished on 4000/11066 queries. users per second: 2511.66"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 5000/11066 queries. users per second: 2480.66</pre>"
      ],
      "text/plain": [
       "recommendations finished on 5000/11066 queries. users per second: 2480.66"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 6000/11066 queries. users per second: 2494.56</pre>"
      ],
      "text/plain": [
       "recommendations finished on 6000/11066 queries. users per second: 2494.56"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 7000/11066 queries. users per second: 2505.28</pre>"
      ],
      "text/plain": [
       "recommendations finished on 7000/11066 queries. users per second: 2505.28"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 8000/11066 queries. users per second: 2506.55</pre>"
      ],
      "text/plain": [
       "recommendations finished on 8000/11066 queries. users per second: 2506.55"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 9000/11066 queries. users per second: 2512.24</pre>"
      ],
      "text/plain": [
       "recommendations finished on 9000/11066 queries. users per second: 2512.24"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 10000/11066 queries. users per second: 2517.17</pre>"
      ],
      "text/plain": [
       "recommendations finished on 10000/11066 queries. users per second: 2517.17"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre>recommendations finished on 11000/11066 queries. users per second: 2525.05</pre>"
      ],
      "text/plain": [
       "recommendations finished on 11000/11066 queries. users per second: 2525.05"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "recom = model1.recommend(users=customers_to_rec, k=8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "3b78af97",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-------------------------------+------------+----------------------+------+\n",
      "|          customer_id          | article_id |        score         | rank |\n",
      "+-------------------------------+------------+----------------------+------+\n",
      "| 000fb6e772c5d0023892065e65... | 610776002  | 0.009831979392487325 |  1   |\n",
      "| 000fb6e772c5d0023892065e65... | 706016001  | 0.008889484847033466 |  2   |\n",
      "| 000fb6e772c5d0023892065e65... | 706016002  | 0.006214228971504871 |  3   |\n",
      "| 000fb6e772c5d0023892065e65... | 782760013  | 0.005852978023481958 |  4   |\n",
      "| 000fb6e772c5d0023892065e65... | 749175001  | 0.005852978023481958 |  5   |\n",
      "| 000fb6e772c5d0023892065e65... | 746298001  | 0.005852978023481958 |  6   |\n",
      "| 000fb6e772c5d0023892065e65... | 714083008  | 0.005852978023481958 |  7   |\n",
      "| 000fb6e772c5d0023892065e65... | 680757003  | 0.005852978023481958 |  8   |\n",
      "| 0024dea548c64fb75a563e0b30... | 706016001  | 0.008689361836265596 |  1   |\n",
      "| 0024dea548c64fb75a563e0b30... | 562245001  | 0.008211719120009262 |  2   |\n",
      "+-------------------------------+------------+----------------------+------+\n",
      "[88528 rows x 4 columns]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "recom.print_rows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "c4bcf960",
   "metadata": {},
   "outputs": [],
   "source": [
    "df3 = tc.SFrame(recom)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "456574e8",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df3 = pd.DataFrame(df3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "0ef9aa22",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>customer_id</th>\n",
       "      <th>article_id</th>\n",
       "      <th>score</th>\n",
       "      <th>rank</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>610776002</td>\n",
       "      <td>0.009832</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>706016001</td>\n",
       "      <td>0.008889</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>706016002</td>\n",
       "      <td>0.006214</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>782760013</td>\n",
       "      <td>0.005853</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>749175001</td>\n",
       "      <td>0.005853</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88523</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>610776001</td>\n",
       "      <td>0.032641</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88524</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>562245046</td>\n",
       "      <td>0.030337</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88525</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>706016003</td>\n",
       "      <td>0.029127</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88526</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>562245001</td>\n",
       "      <td>0.027814</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88527</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>554450001</td>\n",
       "      <td>0.023521</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>88528 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             customer_id  article_id  \\\n",
       "0      000fb6e772c5d0023892065e659963da90b1866035558e...   610776002   \n",
       "1      000fb6e772c5d0023892065e659963da90b1866035558e...   706016001   \n",
       "2      000fb6e772c5d0023892065e659963da90b1866035558e...   706016002   \n",
       "3      000fb6e772c5d0023892065e659963da90b1866035558e...   782760013   \n",
       "4      000fb6e772c5d0023892065e659963da90b1866035558e...   749175001   \n",
       "...                                                  ...         ...   \n",
       "88523  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   610776001   \n",
       "88524  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   562245046   \n",
       "88525  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   706016003   \n",
       "88526  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   562245001   \n",
       "88527  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   554450001   \n",
       "\n",
       "          score  rank  \n",
       "0      0.009832     1  \n",
       "1      0.008889     2  \n",
       "2      0.006214     3  \n",
       "3      0.005853     4  \n",
       "4      0.005853     5  \n",
       "...         ...   ...  \n",
       "88523  0.032641     4  \n",
       "88524  0.030337     5  \n",
       "88525  0.029127     6  \n",
       "88526  0.027814     7  \n",
       "88527  0.023521     8  \n",
       "\n",
       "[88528 rows x 4 columns]"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "cdfda7a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_top_8_articles(group):\n",
    "    return group['article_id'].unique()[:8]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "3617e197",
   "metadata": {},
   "outputs": [],
   "source": [
    "grouped = df3.groupby('customer_id').apply(get_top_8_articles).reset_index()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "07530a21",
   "metadata": {},
   "outputs": [],
   "source": [
    "grouped = grouped.rename(columns={0: 'article_ids'})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "3ff9dea2",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>customer_id</th>\n",
       "      <th>article_ids</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>000fb6e772c5d0023892065e659963da90b1866035558e...</td>\n",
       "      <td>[610776002, 706016001, 706016002, 782760013, 7...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0024dea548c64fb75a563e0b300c0b16210decee446f1a...</td>\n",
       "      <td>[706016001, 562245001, 841992002, 706388002, 5...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>00357b192b81fc83261a45be87f5f3d59112db7d117513...</td>\n",
       "      <td>[610776002, 706016001, 751674001, 905492002, 9...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0036a44bd648ce2dbc32688a465b9628b7a78395302f26...</td>\n",
       "      <td>[448509014, 706016002, 759871002, 706016003, 7...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0040e2fc2d1e7931a38355aca56b2c62b87e65051b7287...</td>\n",
       "      <td>[706016001, 610776002, 759871002, 706016002, 4...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11061</th>\n",
       "      <td>ffddc52a24cd9e170570b48773779eec8ad05bd0cf8163...</td>\n",
       "      <td>[706016001, 610776001, 706016002, 706016003, 4...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11062</th>\n",
       "      <td>ffe6376eb6b854d842e5a7714ea758de127f086a60d67d...</td>\n",
       "      <td>[706016001, 706016002, 871005003, 448509014, 6...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11063</th>\n",
       "      <td>fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e5...</td>\n",
       "      <td>[706016001, 706016002, 448509014, 610776002, 7...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11064</th>\n",
       "      <td>fffae8eb3a282d8c43c77dd2ca0621703b71e90904dfde...</td>\n",
       "      <td>[706016002, 610776002, 706016003, 562245001, 7...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11065</th>\n",
       "      <td>fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...</td>\n",
       "      <td>[706016001, 610776002, 706016002, 610776001, 5...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>11066 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                             customer_id  \\\n",
       "0      000fb6e772c5d0023892065e659963da90b1866035558e...   \n",
       "1      0024dea548c64fb75a563e0b300c0b16210decee446f1a...   \n",
       "2      00357b192b81fc83261a45be87f5f3d59112db7d117513...   \n",
       "3      0036a44bd648ce2dbc32688a465b9628b7a78395302f26...   \n",
       "4      0040e2fc2d1e7931a38355aca56b2c62b87e65051b7287...   \n",
       "...                                                  ...   \n",
       "11061  ffddc52a24cd9e170570b48773779eec8ad05bd0cf8163...   \n",
       "11062  ffe6376eb6b854d842e5a7714ea758de127f086a60d67d...   \n",
       "11063  fff4d3a8b1f3b60af93e78c30a7cb4cf75edaf2590d3e5...   \n",
       "11064  fffae8eb3a282d8c43c77dd2ca0621703b71e90904dfde...   \n",
       "11065  fffb68e203e88449a1dc7173e938b1b3e91b0c93ff4e1d...   \n",
       "\n",
       "                                             article_ids  \n",
       "0      [610776002, 706016001, 706016002, 782760013, 7...  \n",
       "1      [706016001, 562245001, 841992002, 706388002, 5...  \n",
       "2      [610776002, 706016001, 751674001, 905492002, 9...  \n",
       "3      [448509014, 706016002, 759871002, 706016003, 7...  \n",
       "4      [706016001, 610776002, 759871002, 706016002, 4...  \n",
       "...                                                  ...  \n",
       "11061  [706016001, 610776001, 706016002, 706016003, 4...  \n",
       "11062  [706016001, 706016002, 871005003, 448509014, 6...  \n",
       "11063  [706016001, 706016002, 448509014, 610776002, 7...  \n",
       "11064  [706016002, 610776002, 706016003, 562245001, 7...  \n",
       "11065  [706016001, 610776002, 706016002, 610776001, 5...  \n",
       "\n",
       "[11066 rows x 2 columns]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "grouped"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "2f66008d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'000fb6e772c5d0023892065e659963da90b1866035558ec16fca51b0dcfb7e59'"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[0,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "b5a9c55e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Customer Id: 000fb6e772c5d0023892065e659963da90b1866035558ec16fca51b0dcfb7e59\n"
     ]
    },
    {
     "ename": "IndexError",
     "evalue": "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[73], line 4\u001b[0m\n\u001b[1;32m      1\u001b[0m grouped\u001b[38;5;241m.\u001b[39mto_csv(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m../notebooks/dataframe.csv\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m      3\u001b[0m user_customer_id \u001b[38;5;241m=\u001b[39m \u001b[38;5;28minput\u001b[39m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mCustomer Id: \u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m----> 4\u001b[0m recommendation_index \u001b[38;5;241m=\u001b[39m \u001b[43mdf\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mindex\u001b[49m\u001b[43m[\u001b[49m\u001b[43muser_customer_id\u001b[49m\u001b[43m]\u001b[49m\n\u001b[1;32m      5\u001b[0m recommendations \u001b[38;5;241m=\u001b[39m df\u001b[38;5;241m.\u001b[39miloc[recommendation_index, \u001b[38;5;241m2\u001b[39m]\n\u001b[1;32m      7\u001b[0m recommendations\n",
      "File \u001b[0;32m~/opt/anaconda3/envs/bigdata_ml/lib/python3.8/site-packages/pandas/core/indexes/range.py:972\u001b[0m, in \u001b[0;36mRangeIndex.__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m    968\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mIndexError\u001b[39;00m(\n\u001b[1;32m    969\u001b[0m             \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mindex \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mkey\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m is out of bounds for axis 0 with size \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mlen\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    970\u001b[0m         ) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01merr\u001b[39;00m\n\u001b[1;32m    971\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m is_scalar(key):\n\u001b[0;32m--> 972\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mIndexError\u001b[39;00m(\n\u001b[1;32m    973\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124monly integers, slices (`:`), \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    974\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mellipsis (`...`), numpy.newaxis (`None`) \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    975\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mand integer or boolean \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    976\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124marrays are valid indices\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    977\u001b[0m     )\n\u001b[1;32m    978\u001b[0m \u001b[38;5;66;03m# fall back to Int64Index\u001b[39;00m\n\u001b[1;32m    979\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28msuper\u001b[39m()\u001b[38;5;241m.\u001b[39m\u001b[38;5;21m__getitem__\u001b[39m(key)\n",
      "\u001b[0;31mIndexError\u001b[0m: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"
     ]
    }
   ],
   "source": [
    "grouped.to_csv('../notebooks/dataframe.csv')\n",
    "\n",
    "user_customer_id = input('Customer Id: ')\n",
    "recommendation_index = df.index[user_customer_id]\n",
    "recommendations = df.iloc[recommendation_index, 2]\n",
    "\n",
    "recommendations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "3e0bbdf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_article_ids(df, customer_id):\n",
    "    article_ids = df[df['customer_id'] == customer_id]['article_id'].tolist()\n",
    "    return article_ids"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "34ae5a37",
   "metadata": {},
   "outputs": [],
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
    "        article_ids = get_article_ids(df, customer_id)\n",
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
   "id": "95620bd9",
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
