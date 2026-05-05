import streamlit as st
import requests
import base64
import os

st.set_page_config(page_title="Dynamic ML Visualizer", page_icon="🔬")

st.title("🔬 Dynamic ML Visualizer")
st.markdown("Swap datasets on the fly. PyTorch dynamically adjusts its input architecture.")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state to track the active dataset
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'data_info' not in st.session_state:
    st.session_state.data_info = None

# 1. Fetch available datasets
try:
    datasets_res = requests.get(f"{API_URL}/datasets", timeout=5)
    available_datasets = datasets_res.json().get("available_datasets", [])
except:
    available_datasets = []

if not available_datasets:
    st.error("🚨 Could not connect to the ML API.")
else:
    # 2. Dataset Dropdown
    selected_dataset = st.selectbox("📁 Select a Dataset:", available_datasets)

    # 3. If the user changed the dropdown, fetch the new dataset from the API
    if selected_dataset != st.session_state.current_dataset:
        with st.spinner(f"Loading {selected_dataset} data..."):
            try:
                res = requests.post(f"{API_URL}/load-dataset", json={"name": selected_dataset}, timeout=10)
                res.raise_for_status()
                st.session_state.data_info = res.json()["data"]
                st.session_state.current_dataset = selected_dataset
                st.success(f"✅ Successfully loaded **{selected_dataset}** ({st.session_state.data_info['train_size']} training samples)")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to load dataset: {e}")

    # 4. Render UI only if data is loaded
    if st.session_state.data_info:
        try:
            pca_res = requests.get(f"{API_URL}/plot/pca", timeout=5)
            st.image(base64.b64decode(pca_res.json()["image_base64"]), caption=f"PCA Feature Space ({st.session_state.current_dataset})")
        except:
            st.warning("Could not load plot.")

        st.divider()

        # 5. Training Configuration
        st.subheader("⚙️ Train Model")
        col1, col2, col3 = st.columns(3)
        with col1: epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
        with col2: lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%4f")
        with col3: hidden = st.number_input("Hidden Dim", min_value=8, max_value=128, value=32)

        if st.button("Train PyTorch Model", type="primary", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    response = requests.post(f"{API_URL}/train", json={"epochs": epochs, "lr": lr, "hidden_dim": hidden}, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                    
                    st.subheader("Training Results")
                    m1, m2 = st.columns(2)
                    m1.metric(label="Test Accuracy", value=f"{data['accuracy']:.2%}")
                    m2.metric(label="Classes", value=f"{st.session_state.data_info['target_names'][0]} vs {st.session_state.data_info['target_names'][1]}")
                    
                    st.image(base64.b64decode(data["training_plot_base64"]), caption="Loss & Accuracy Curves")
                    st.image(base64.b64decode(data["confusion_plot_base64"]), caption="Confusion Matrix Heatmap")
                    st.image(base64.b64decode(data["decision_boundary_base64"]), caption="Decision Boundary in PCA Space")
                    
                except requests.exceptions.RequestException as e:
                    st.error("🚨 Training failed.")