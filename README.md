# Chladni: Grokking Simulation Platform

Chladni is a web-based simulation platform designed to visualize the "grokking" phenomenon in neural networks. The platform trains a custom, small-scale Transformer model on modular arithmetic and streams the learning metrics and token embeddings in real-time to a web dashboard. This allows researchers and engineers to observe the live phase transition from memorization to genuine algorithmic generalization.

## Architecture

The project is structured into two main components: a Python-based machine learning backend and a frontend web dashboard built with Next.js.

### Backend

The backend is responsible for data generation, model training, and real-time data streaming.

*   **Model**: A custom 1-layer PyTorch Transformer with approximately 143k parameters. The reduced size ensures rapid CPU-based training capable of streaming at interactive framerates.
*   **Dataset**: Generates exhaustive examples of modular arithmetic, specifically `(a + b) mod p` where `p` is 97 by default.
*   **API**: A FastAPI web server that provides a core WebSocket endpoint for live training data and HTTP endpoints for structured batch data.
*   **Streaming**: The WebSocket streams the current step, training loss, training accuracy, test accuracy, and the dimensionality-reduced (PCA) coordinates of the token embeddings every 100 steps.

### Frontend Dashboard

The frontend is a React application built with Next.js 15, TypeScript, and TailwindCSS. It provides a real-time, interactive window into the model's internal representations.

*   **Simulation Controls**: Adjust dynamic hyperparameters such as Dataset Fraction (which drives generalization) and Weight Decay (which provides the structural pressure needed for grokking).
*   **Accuracy Curves**: Real-time line charts tracking the decoupling of training and test accuracy using Recharts.
*   **Token Embedding Space**: A live 3D scatter plot powered by Three.js. As the model trains, it visualizes the geometric structures formed within the embeddings across different modular operations, dynamically projecting onto an L2-normalized sphere to prevent magnitude collapse. Includes geometric angular sorting to cleanly render the complex Fourier-basis harmonic structures (rings/toruses) that emerge during mathematical generalization.
*   **Early Warning System**: A live mathematical monitor running alongside training. It continuously computes the discrete Fourier transform of the embeddings to measure feature emergence and uses a Logit-transformed regression model applied to the testing accuracy's Sigmoid S-curve to algorithmically predict exactly which step grokking will occur.

## Installation

### Prerequisites

*   Python 3.10+
*   Node.js 18+

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/ayoitssmit/Chladni.git
    cd Chladni
    ```

2.  Initialize the Backend:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

3.  Initialize the Frontend:
    ```bash
    cd ../frontend
    npm install
    ```

## Usage

### Running the Platform

The platform requires both the backend API and the frontend client to be running simultaneously.

1.  Start the FastAPI Server:
    ```bash
    cd backend
    python server.py
    ```
    The server listens on `http://localhost:8000`.

2.  Start the Next.js Client:
    ```bash
    cd frontend
    npm run dev
    ```
    The dashboard will be available at `http://localhost:3000`.



## License

MIT
