import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# 1. Define the Class for text preprocessing (used in the pipeline saved in the pickle file)
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words, lemmatizer):
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        processed_texts = []
        for text in X:
            text = re.sub('[^a-zA-Z]', ' ', text)
            text = text.lower()
            text_words = text.split()
            text_words = [word for word in text_words if word not in self.stop_words]
            text_words = [self.lemmatizer.lemmatize(word) for word in text_words]
            processed_texts.append(' '.join(text_words))
        return processed_texts

# 2. Load the pipeline and vectorized data from the pickle file
with open('model_pipeline.pkl', 'rb') as file:
    pipeline, data_vectors, data = pickle.load(file)


# 3. Input from the user
input_text = "I like wines that are smooth and have a hint of spice"

# 4. Process the input text using the loaded pipeline (The pipeline includes both preprocessing and vectorization)
input_vector = pipeline.transform([input_text])

# 5. Calculate cosine similarity between the input text vector and all data_vectors
cosine_sim_matrix = cosine_similarity(input_vector, data_vectors)

# 6. Find the indices of the top 5 recommendations based on cosine similarity
top_5_indices = np.argsort(-cosine_sim_matrix, axis=1)[:, :5][0]

# 7. Retrieve the top 5 recommendations from the original data
recommendations = data.iloc[top_5_indices]

# 8. Print the information about the top 5 wine recommendations (can add other fields also as per the frontend design)
print(recommendations[['title', 'description']])