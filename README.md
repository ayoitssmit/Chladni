# Chladni: Grokking Simulation Platform

Chladni is a web-based simulation platform designed to visualize the "grokking" phenomenon in neural networks. The platform trains a custom, small-scale Transformer model on modular arithmetic and streams the learning metrics and token embeddings in real-time to a web dashboard. This allows for the live observation of the phase transition from memorization to generalization.

## Architecture

The project is structured into two main components: a Python-based machine learning backend and a frontend web dashboard (currently in development).

### Backend

The backend is responsible for data generation, model training, and real-time data streaming.

*   **Model**: A custom 1-layer PyTorch Transformer with approximately 143k parameters. The small size allows for rapid CPU-based training.
*   **Dataset**: Generates exhaustive examples of modular addition, specifically `(a + b) mod p` where `p` is 97 by default.
*   **API**: A FastAPI web server that provides a health check and the core WebSocket endpoint for live training data.
*   **Streaming**: The WebSocket streams the current step, training loss, training accuracy, test accuracy, and the 3D PCA coordinates of the token embeddings every 100 steps.

### Frontend (To Be Developed)

The frontend will be a React application built with TypeScript, Vite, and TailwindCSS. It will feature:
*   A control panel for adjusting hyperparameters like dataset fraction and weight decay.
*   A real-time line chart tracking training and test accuracy using Recharts.
*   A 3D scatter plot of the token embeddings using Three.js to visualize the geometric structures formed during grokking.

## Installation

### Prerequisites

*   Python 3.10+
*   pip

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/ayoitssmit/Chladni.git
    cd Chladni
    ```

2.  Navigate to the backend directory and install the requirements:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

## Usage

### Running the API Server

Start the FastAPI server:
```bash
python server.py
```
The server will start on `http://localhost:8000`. 
WebSocket Endpoint: `ws://localhost:8000/ws/simulate`

### Testing the WebSocket Stream

With the server running, open a new terminal and run the test client:
```bash
python test_ws.py
```
This will connect to the server, start a training run, and print the first few streamed snapshots of accuracy and PCA embeddings.

### Generating the Phase Diagram

To run a batch of training simulations across a grid of dataset fractions and weight decays:
```bash
python batch_run.py
```
This script will produce a `phase_diagram.json` file containing the step at which grokking occurred for each combination of parameters.

## License

MIT
