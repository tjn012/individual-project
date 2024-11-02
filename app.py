import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Path to the dataset file
file_path = 'my_text_file.txt'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found in the environment.")

# Load the dataset
with open(file_path, 'r') as f:
    data = f.readlines()

# Ensure the data is loaded correctly
if not data:
    raise ValueError("The dataset is empty or could not be loaded properly.")

# Preprocessing function for text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer=preprocess_text)
tfidf_matrix = vectorizer.fit_transform(data)

# Chatbot response function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix)
    most_similar_index = cosine_similarities.argmax()
    return data[most_similar_index].strip()

# Gradio interface
def chatbot_interface(user_input):
    response = chatbot_response(user_input)
    return response

# Create a Gradio interface for the chatbot
iface = gr.Interface(fn=chatbot_interface, 
                     inputs="text", 
                     outputs="text",
                     title="FAQ Chatbot",
                     description="Ask a question to the FAQ chatbot.")

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
