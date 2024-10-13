import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model_path = r"C:\Users\ASUS\Music\MLDL_projects\Models\BERT_ft_epoch1"
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Create the text classification pipeline
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# Label mapping
label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Positive'}

# Streamlit application
st.title("BERT Sentiment Classifier")

# Input sentence
sentence = st.text_input("Enter a sentence to classify:")

if sentence:
    # Get the result from the classifier
    result = classifier(sentence)

    # Display the prediction
    predicted_label = label_map[result[0]['label']]
    score = round(result[0]['score'], 3)
    
    st.write(f"**Predicted label:** {predicted_label}")
    st.write(f"**Confidence Score:** {score}")
