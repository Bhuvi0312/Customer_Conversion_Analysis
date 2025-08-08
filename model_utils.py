{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "85b20902-f70b-4ee8-9a7c-7db429a31e8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Libraies for Modern Pipeline\n",
    "from sklearn.linear_model import LogisticRegression, LinearRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import classification_report, mean_squared_error, silhouette_score\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a8b94816-805f-40f2-bbc7-45d3a91066bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Model Pipeline\n",
    "def get_classification_pln():\n",
    "    return pipeline(steps=[\n",
    "        ('preprocessing',data_preprocessing),\n",
    "        ('classifier',RandomForestClassifier(class_weight='balanced'))\n",
    "    ])\n",
    "\n",
    "def get_regression_pln():\n",
    "    return pipeline(steps=[\n",
    "        ('preprocessing',data_preprocessing),\n",
    "        ('regression',LinearRegression())\n",
    "    ])\n",
    "\n",
    "def get_clustering_pln(n_clusters=3):\n",
    "    return pipeline(steps=[\n",
    "        ('preprocessing',data_preprocessing),\n",
    "        ('clustering',KMeans(n_clusters=n_clusters))\n",
    "    ])\n",
    "\n",
    "def evluate_classification(y_true,y_pred):\n",
    "    return classification_report(y_true,y_pred,output_dic=True)\n",
    "\n",
    "def evaluate_regression(y_true,y_pred):\n",
    "    mse = mean_squared_error(y_true,y_pred)\n",
    "    return {'MSE':mse,'RMSE':mse**0.5}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
