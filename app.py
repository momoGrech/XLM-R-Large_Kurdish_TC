import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

torch.classes.__path__ = []

repo_id = "MomoGrech/XLM-R-Large_Kurdish_Sorani_Text_Classification"

@st.cache_resource
def get_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer

model,tokenizer=get_model(repo_id)

# st.write("Text Classification in the Kurdish Sorani Language using fine-tuned XLM-RoBERTa Model")
st.markdown("""
    <h2 style='text-align: center; color: #3b3b3b;'>
        Text Classification in the Kurdish Sorani Language Using Fine-Tuned XLM-RoBERTa Model
    </h2>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter Kurdish Text to classify",
    height=200,
    key="text"
) 

def clear_text():
  st.session_state["text"] = ''  

def clear_text():
    st.session_state["text"] = ''

col1, col2 = st.columns([1, 1]) 

with col1:
    classifyButton = st.button("Classify Text")

with col2:
    st.button("Clear Text", on_click=clear_text)


dict = {0: 'Sport', 1: 'Health', 2: 'Science & technology', 3: 'Social', 4: 'Economic'}

label_df = pd.DataFrame({
    "Label ID": [0, 1, 2, 3, 4],
    "Category": ['Sport', 'Health', 'Science & Technology', 'Social', 'Economic']
})

st.markdown("""
            <style>
                div[data-testid="stColumn"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="stColumn"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

 
files = st.file_uploader("Upload a file", type=["csv", "txt"], accept_multiple_files=True)
 
# Read files and keep their names
@st.cache_data
def read_uploaded_files(files):
    return [(file.name, file.read().decode("utf-8")) for file in files]


texts = read_uploaded_files(files)

# Trigger classification when button is pressed
if texts and classifyButton:
    for file_name, text in texts:
        test_sample = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        output = model(**test_sample)
        y_pred = np.argmax(output.logits.detach().cpu().numpy(), axis=1)
        prediction = dict[y_pred[0]]

        st.markdown(
            f"<h4>📄 <span style='color: black;'>{file_name}</span></h4>"
            f"<h5><span style='color: black;'>Predicted subject: </span> "
            f"<span style='color: green;'>{prediction}</span></h5>",
            unsafe_allow_html=True
        )

        # Extract weights and round
        weights_list = output.logits.detach().cpu().numpy().tolist()[0]
        rounded_weights = [round(w, 4) for w in weights_list]

        # Find the max weight
        max_weight = max(rounded_weights)

        # Add formatted weights to label_df
        formatted_weights = []
        for w in rounded_weights:
            if w == max_weight:
                formatted_weights.append(f"<b><span style='color: green;'>{w}</span></b>")
            else:
                formatted_weights.append(str(w))

        # Create a copy of the dataframe with formatted weights
        styled_df = label_df.copy()
        styled_df["Weight"] = formatted_weights

        # Convert to HTML table
        html_table = styled_df.to_html(escape=False, index=False, justify='start')

        # Display
        st.markdown("""
            <style>
                .stTable tr {
                    height: 50px; # use this to adjust the height
                }
            </style>
        """, unsafe_allow_html=True)
        st.markdown(html_table, unsafe_allow_html=True)



# Trigger classification when button is pressed
if user_input and classifyButton:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors="pt")
    output = model(**test_sample)
    y_pred = np.argmax(output.logits.detach().cpu().numpy(), axis=1)

    prediction = dict[y_pred[0]]
    st.markdown(
        f"<h3><span style='color: black;'>The predicted subject is: </span> "
        f"<span style='color: green;'>{prediction}</span></h3>",
        unsafe_allow_html=True
    )

    # Extract weights, round them
    weights_list = output.logits.detach().cpu().numpy().tolist()[0]
    rounded_weights = [round(w, 4) for w in weights_list]

    # Add the weights as a new column to the DataFrame
    label_df["Weight"] = rounded_weights

    st.write("**Label Dictionary with Coresponding Weights:**")
    st.table(label_df)#