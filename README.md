### 🔬 Cancer Prediction Visualizer

An interactive, containerized machine learning application that trains a PyTorch neural network on the Breast Cancer Wisconsin dataset and visualizes the outcomes using Matplotlib. 

Built with a decoupled microservice architecture: a FastAPI backend handles the heavy ML computations and image rendering, while a Streamlit frontend serves as the interactive web dashboard.

### 📁 Project Structure
```
cancer-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml      
├── Dockerfile.api          
├── Dockerfile.ui           
├── requirements.txt        
├── src/
│   ├── model.py            
│   ├── api.py              
│   └── ui/
│       └── ui.py           
└── README.md
```

### ⚙️ Tech Stack
```
Language: Python 3.10
API Framework: FastAPI + Uvicorn
Deep Learning: PyTorch (Feedforward Neural Network)
Data Processing: NumPy, Pandas, Scikit-learn (PCA, StandardScaler, SVM)
Visualization: Matplotlib (Agg backend for containerized base64 encoding)
Frontend: Streamlit
Containerization: Docker
Orchestration: Docker Compose
CI/CD GitHub Actions
```

###🚀 Quick Start 
Prerequisites 

     Docker Desktop  installed and running.
     Python 3.10+ installed
     

Run the Application 

Clone or download this repository. 
Open a terminal in the project root directory. 
Run the following command:
bash
     
```
     
git clone https://github.com/LorexGitHub/cancer-predictor.git
cd cancer-predictor
docker compose up --build
     
```
      
Wait for the terminal to display: You can now view your Streamlit app in your browser. 
Open your browser and navigate to: http://localhost:8501  

(To stop the application, press Ctrl+C in your terminal). 