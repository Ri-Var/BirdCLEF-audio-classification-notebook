# BirdCLEF Audio Classification Notebook

This repository contains a notebook for bird sound classification using deep learning techniques and audio preprocessing pipelines. The project is built around the [BirdCLEF](https://www.kaggle.com/competitions/birdclef-2023) dataset and aims to classify 206 different bird species from audio recordings.

---

## 📌 Project Highlights

- 📁 **Audio Preprocessing**: 
  - Noise reduction using `noisereduce`
  - Voice Activity Detection (VAD) using Silero VAD
  - Conversion to Mel spectrograms

- 🧠 **Modeling**:
  - Feature extraction using pre-trained [YAMNet](https://tfhub.dev/google/yamnet/1)
  - Feedforward Neural Network with PyTorch
  - Class balancing via `class_weight`

- 🧪 **Evaluation**:
  - Accuracy, classification report, and recall score
  - Train-test split used for validation

---

## 📂 File Structure

- `Classifier.ipynb`: Main notebook with preprocessing, modeling, and evaluation.
- `README.md`: Project overview and instructions.

---

## ⚙️ Requirements

Make sure the following packages are installed:

```bash
pip install pandas numpy matplotlib torch torchaudio noisereduce scikit-learn tensorflow_hub
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Ri-Var/BirdCLEF-audio-classification-notebook.git
   cd BirdCLEF-audio-classification-notebook
   ```

2. Open `Classifier.ipynb` in Jupyter Notebook or VSCode.

3. Run all cells sequentially. Ensure internet access for downloading the YAMNet model and Silero VAD.

---

## Results

The model is evaluated on a train-test split using metrics like:
- Accuracy
- Recall
- Precision
- Classification Report

---
## References

- [BirdCLEF Challenge](https://www.kaggle.com/competitions/birdclef-2025)
- [YAMNet (TensorFlow Hub)](https://tfhub.dev/google/yamnet/1)
- [Silero VAD](https://github.com/snakers4/silero-vad)

---

## Contact

**Ritish Varada**  
Email: [vrtish06@gmail.com](mailto:vrtish06@gmail.com)  
GitHub: [@Ri-Var](https://github.com/Ri-Var)
