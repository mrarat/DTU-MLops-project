import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2


def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    response = requests.post(predict_url, files={"image": image}, timeout=10)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = "placeholder"  # get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Playing Cards Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        # result = classify_image(image, backend=backend)
        result = {
            "prediction": "Ace of Spades",
            "probabilities": [0.1, 0.05, 0.1, 0.6, 0.05, 0.02, 0.03, 0.02, 0.01, 0.02],
        }

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # Show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.header(f"Prediction: {prediction}")

            # Rank probabilities bar chart
            data_rank = {"Rank": [f"Rank {i}" for i in range(10)], "Probability": probabilities}
            df_rank = pd.DataFrame(data_rank)
            df_rank.set_index("Rank", inplace=True)
            st.bar_chart(df_rank, y="Probability")

            # Suit probabilities bar chart
            data_suit = {"Suit": [f"Suit {i}" for i in range(10)], "Probability": probabilities[::-1]}
            df_suit = pd.DataFrame(data_suit)
            df_suit.set_index("Suit", inplace=True)
            st.bar_chart(df_suit, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
