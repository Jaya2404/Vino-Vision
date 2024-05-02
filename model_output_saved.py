import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# 1. Define the Class for text preprocessing (used in the pipeline saved in the pickle file)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words, stemmer):
        self.stop_words = stop_words
        self.stemmer = stemmer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        processed_texts = []
        for text in X:
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text_words = text.split()
            text_words = [word for word in text_words if word not in self.stop_words]
            text_words = [self.stemmer.stem(word) for word in text_words]
            processed_texts.append(' '.join(text_words))
        return processed_texts


# Load the saved components
with open('model_components.pkl', 'rb') as file:
    pipeline, data, multilabel_binarizer = pickle.load(file)


# 3. Input from the user
user_input_description = "I like wines that are smooth and have a hint of spice"

# Making prediction
y_user_pred_proba = pipeline.predict_proba([user_input_description])  # Correct use of predict_proba

# Get the indices of the probabilities sorted in descending order
sorted_indices = np.argsort(-y_user_pred_proba, axis=1)[0]

# Initialize a set to keep track of unique varieties recommended
unique_varieties = set()
unique_recommendations = []

# Loop over sorted indices and get unique varieties
for idx in sorted_indices:
    variety = data.iloc[idx]['variety']
    if variety not in unique_varieties:
        unique_varieties.add(variety)
        unique_recommendations.append(data.iloc[idx])

        # Break the loop if we have enough recommendations
        if len(unique_recommendations) == 5:
            break

# Convert the list of unique recommendations to a DataFrame for easy viewing
unique_recommendations_df = pd.DataFrame(unique_recommendations)

# Print the top 5 unique recommended varieties along with other details
print("Top 5 unique recommended wines based on your description:")
print(unique_recommendations_df[['variety', 'description', 'price']])

