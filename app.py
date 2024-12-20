import streamlit as st
import pickle
import re
import nltk
import PyPDF2
import docx

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained models and encoders
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))  # Label encoder for categories

# Function to clean resume text
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)  # Remove URLs
    text = re.sub(r'@\S+', ' ', text)    # Remove mentions
    text = re.sub(r'#\S+', ' ', text)    # Remove hashtags
    text = re.sub(r'RT|cc\S+', ' ', text)  # Remove retweets
    text = re.sub(r"[\"!#$%^&*()+_={}\[\]:;'<>,.?/|\\`~]", ' ', text)  # Remove special characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text.lower()  # Convert to lowercase

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Function to extract text from TXT
def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Function to predict the category of a resume
def predict_category(resume_text):
    cleaned_text = clean_resume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)
    return le.inverse_transform(prediction)[0]  # Return the category name

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Text successfully extracted from the uploaded resume!")

            if st.checkbox("Show Extracted Text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Job Category")
            category = predict_category(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

if __name__ == "__main__":
    main()
