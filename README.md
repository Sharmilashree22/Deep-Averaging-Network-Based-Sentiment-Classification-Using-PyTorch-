
🧠 Deep Averaging Network-Based Sentiment Classifier (PyTorch Implementation)
This project presents an implementation of a Deep Averaging Network (DAN) for binary sentiment classification using the CFIMDB IMDb review dataset. It explores neural text classification through modular, extensible PyTorch components, with a focus on embedding layer construction, feedforward architecture design, and efficient batching via sequence padding.

📁 Project Structure
bash
Copy
Edit
UID/
├── main.py                   # Training and evaluation script (pad_sentences implemented)
├── model.py                  # DAN model architecture with parameter initialization
├── vocab.py                  # Vocabulary construction utility
├── setup.py                  # Optional pre-processing 
├── data/
│   ├── cfimdb-train.txt      # Training data
│   ├── cfimdb-dev.txt        # Dev set with labels
│   ├── cfimdb-test.txt       # Test set with hidden labels
├── cfimdb-dev-output.txt     # Dev set predictions
├── cfimdb-test-output.txt    # Test set predictions

🚀 Setup Instructions
1. Environment Setup
Ensure Python ≥3.8 is installed. The only allowed external libraries are:

numpy

torch (PyTorch ≥1.10)

You may optionally use Google Colab for GPU acceleration.

bash
Copy
Edit
pip install numpy torch
2. Embedding Preparation (Optional)
If using pre-trained word embeddings (e.g., GloVe):

Download from GloVe official site

Save the .txt file

Modify setup.py to automate loading

Reference the file path via --emb_file argument in main.py

3. Model Implementation Tasks
model.py
define_model_parameters(): Constructs the architecture including:

nn.Embedding

One or more nn.Linear layers

Dropout + activation functions (ReLU/Tanh)

init_model_parameters(): Initializes weights using Xavier/Glorot or uniform methods for stable convergence.

load_embedding(): Parses pre-trained embeddings and maps vocab tokens to vectors.

copy_embedding_from_numpy(): Transfers pre-trained numpy embeddings to PyTorch tensors.

main.py
pad_sentences(): Implements dynamic padding for batch-wise sentence processing. Ensures equal-length input by appending PAD_ID.


Make sure run_exp.sh includes default hyperparameters.

📈 Expected Results
With proper implementation and default settings (PyTorch 1.10.2, numpy 1.21.1), you should observe:

Dev Accuracy ≈ 92.24%

Test set predictions saved in cfimdb-test-output.txt (labels in test set are hidden for blind grading)



🧠 References
Iyyer et al., Deep Unordered Composition Rivals Syntactic Methods for Text Classification (ACL 2015): Link

IMDb Dataset: OpenReview CFIMDB Dataset
