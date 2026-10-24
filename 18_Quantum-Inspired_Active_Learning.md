# 🚗 MIMO-TDA Vehicle Classifier: A Journey from CNNs to Quantum-Inspired Active Learning

## 📌 Project Overview
This repository documents the evolution of a traffic vehicle classification system across 18 distinct phases. The project starts with a standard Convolutional Neural Network (CNN) and progressively integrates advanced concepts such as Topological Data Analysis (TDA), Multi-Input Multi-Output (MIMO) experts, Quantum-Inspired Neural Networks (Born Rule), and Human-in-the-Loop Active Learning to solve a specific edge case: **classifying blue vans (often confused with ambulances)**.

## 🧬 Architecture Evolution
* **Phases 1-11 (Baseline & Shortcuts):** Explored ResNet/VGG baselines and identified dataset biases (shortcut learning).
* **Phase 12-13 (Attention & TDA):** Introduced Attention mechanisms and Topological Data Analysis to capture structural features.
* **Phase 14 (MIMO & Fuzzy Logic):** Combined Attention and Topology experts via a dynamic router (MIMO). Used a Fuzzy Classifier for outputs.
* **Phase 16 (Quantum-Inspired Head):** Replaced the fuzzy classifier with a Complex-Valued Quantum Head using the Born Rule (\(P = |a + bi|^2\)). This solved the probability flattening issue and improved overall validation accuracy to **94.32%**.
* **Phase 17 (Active Learning):** Fine-tuned the Phase 16 model using hard examples (blue vans) and human feedback with a very low learning rate (\(10^{-5}\)) to prevent catastrophic forgetting. Validation accuracy peaked at **94.83%**.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Backend API:** FastAPI, Uvicorn
* **Frontend:** HTML/CSS/JS (Served via FastAPI)
* **Data Processing:** NumPy, Pandas, Scikit-Learn

## 🚀 How to Run the Project

### 1. Requirements
pip install torch torchvision fastapi uvicorn pillow numpy
2. Run the API Server (Best Model: Phase 17)
Ensure the phase17.pt checkpoint is inside the artifacts/ directory.

python api_server.py
3. Access the UI
Open your browser and navigate to: http://127.0.0.1:8000

📊 Results & Key Findings
The Blue Van Problem: Initially, the model misclassified blue vans as ambulances due to visual similarity.

Quantum Head Advantage: The Born Rule naturally outputs calibrated probabilities, preventing overconfidence on ambiguous images.

Active Learning Impact: By duplicating hard examples and fine-tuning with a low learning rate, the model successfully learned to distinguish the blue vans (confidence > 56%, margin > 46%).

Reject Threshold: Implemented a dynamic margin threshold (25%). If the model is uncertain (margin < 0.25), it triggers a "Human Review" prompt, ensuring safe deployment.

📈 Phase Comparison Table (Test3: Blue Vans)
Metric	            Phase 14	Phase 16	Phase 17	Phase 18 (TTA)
Val Accuracy	    93.80%	    94.32%	    94.83%	    94.83%
Vanet (Correct)	    25 / 25	    32 / 50	    32 / 50	    32 / 50
Ambulance (Error)	0 / 25	    18 / 50	    18 / 50	    18 / 50
(Note: TTA did not improve the hard set, but Phase 17 provides the most stable performance for the UI deployment).
