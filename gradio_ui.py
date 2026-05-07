import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


def predict(temp):
    response = requests.post(
        API_URL,
        json={"X_fahrenheit": temp}
    )

    result = response.json()
    return result["prediction"][0]


# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Slider(label="Fahrenheit Temperature", minimum=-10, maximum=60 , step=1),
    outputs=gr.Textbox(label="Prediction"),
    title="🌡️ MLflow Temperature Predictor",
    description="Enter a temperature in Fahrenheit and get ML model prediction."
)

iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
