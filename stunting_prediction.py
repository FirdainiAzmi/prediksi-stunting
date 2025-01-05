import pandas as pd
import numpy as np
from math import sqrt, exp, pi
import random
import streamlit as st
import base64

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del summaries[-1] 
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) * 2 / (2 * stdev * 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def load_data():
    file_path = 'Stunting.csv'
    data = pd.read_csv(file_path)

    data['Birth Weight'] = pd.to_numeric(data['Birth Weight'], errors='coerce')
    data['Birth Length'] = pd.to_numeric(data['Birth Length'], errors='coerce')

    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['Stunting'] = data['Stunting'].map({'No': 0, 'Yes': 1})

    data_cleaned = data.dropna()
    return data_cleaned.values.tolist()

def train_test_split(data, test_size=0.2, random_seed=42):
    random.seed(random_seed)
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

data_as_list = load_data()

data_train, data_test = train_test_split(data_as_list, test_size=0.2, random_seed=42)
model = summarize_by_class(data_train)

predictions = [predict(model, row[:-1]) for row in data_test]
actual = [row[-1] for row in data_test]
accuracy = np.mean([pred == act for pred, act in zip(predictions, actual)]) * 100

if "page" not in st.session_state:
    st.session_state["page"] = "home"

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    b64_image = base64.b64encode(image_data).decode()
    bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_image}");
            background-size: cover;
            background-position: center;
        }}
        </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

background_home = "depan ai.png"
background_prediction = "utama ai.png"

if st.session_state["page"] == "home":
    set_background(background_home)
    st.title("Selamat Datang di Aplikasi Prediksi Stunting")
    st.write("Aplikasi ini dirancang untuk membantu memprediksi kemungkinan stunting pada anak berdasarkan data kesehatan.")
    
    if st.button("Lanjutkan"):
        st.session_state["page"] = "prediction"

elif st.session_state["page"] == "prediction":
    set_background(background_prediction)
    st.title("Prediksi Stunting")
    st.write(f"Akurasi model: {accuracy:.2f}%")

    st.subheader("Masukkan Data:")
    gender = st.selectbox("Gender", [("Male", 0), ("Female", 1)], format_func=lambda x: x[0])
    age = st.number_input("Age (in months)", min_value=0.0, max_value=100.0, value=12.0)
    birth_weight = st.number_input("Birth Weight (in kg)", min_value=0.0, value=3.0)
    birth_length = st.number_input("Birth Length (in cm)", min_value=0.0, value=50.0)
    body_weight = st.number_input("Current Body Weight (in kg)", min_value=0.0, value=10.0)
    body_length = st.number_input("Current Body Length (in cm)", min_value=0.0, value=80.0)

    new_data = [gender[1], age, birth_weight, birth_length, body_weight, body_length]

    if st.button("Predict Stunting"):
        predicted_label = predict(model, new_data)
        label_mapping = {0: 'No', 1: 'Yes'}
        st.write(f"Predicted Stunting: {label_mapping[predicted_label]}, lakukan pemeriksaan lebih lanjut ke pusat kesehatan")
