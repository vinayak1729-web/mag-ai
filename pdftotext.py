import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    with open(pdf_file, 'rb') as pdf:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    """Preprocesses text by tokenizing and stemming."""
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def extract_entities(text):
    """Extracts named entities from text."""
    entities = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
    return entities

def classify_intent(text):
    """Classifies the intent of the text (placeholder for actual implementation)."""
    # Replace this with your actual intent classification logic
    return "unknown"

def generate_response(text, pdf_text):
    """Generates a response based on the user's text and the PDF text."""
    # Preprocess both texts
    user_text_processed = preprocess_text(text)
    pdf_text_processed = preprocess_text(pdf_text)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    user_vector = vectorizer.fit_transform([user_text_processed])
    pdf_vector = vectorizer.transform([pdf_text_processed])

    # Calculate cosine similarity
    similarity = cosine_similarity(user_vector, pdf_vector)[0][0]

    if similarity > 0.7:
        # Response based on similarity
        return "I found relevant information in the PDF. Would you like me to summarize it?"
    else:
        # Response indicating lack of relevance
        return "I couldn't find any relevant information in the PDF. Please try rephrasing your query."

# Example usage
pdf_file = "VinayakResume.pdf"
pdf_text = extract_text_from_pdf(pdf_file)

while True:
    user_input = input("Enter your query: ")
    response = generate_response(user_input, pdf_text)
    print(response)