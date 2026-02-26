import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Engine", layout="wide")
st.title("🌳 Logic-Based Decision Tree (ID3)")

data = pd.DataFrame({
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "High", "Normal", "High", "Normal", "High"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
})

with st.expander("View Source Training Data"):
    st.table(data)

def entropy(col):
    counts = np.unique(col, return_counts=True)[1]
    return -sum((c/len(col)) * math.log2(c/len(col)) for c in counts)

def info_gain(df, attr, target):
    total = entropy(df[target])
    vals = df[attr].unique()
    return total - sum((len(df[df[attr]==v])/len(df)) * entropy(df[df[attr]==v][target]) for v in vals)

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]
    best = max(attrs, key=lambda x: info_gain(df, x, target))
    tree = {best: {}}
    for v in df[best].unique():
        tree[best][v] = id3(df[df[best]==v], target, [a for a in attrs if a != best])
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    key = next(iter(tree))
    return predict(tree[key][sample[key]], sample)

col1, col2 = st.columns(2)

with col1:
    if st.button("Build Logic Model"):
        st.session_state.my_tree = id3(data, "PlayTennis", ["Outlook", "Humidity"])
        st.success("Model Synchronized")

if "my_tree" in st.session_state:
    with col2:
        st.subheader("Interactive Prediction")
        out_val = st.selectbox("Environment", data["Outlook"].unique())
        hum_val = st.radio("Moisture Level", data["Humidity"].unique())
        if st.button("Run Inference"):
            res = predict(st.session_state.my_tree, {"Outlook": out_val, "Humidity": hum_val})
            st.metric(label="Decision", value=res)
