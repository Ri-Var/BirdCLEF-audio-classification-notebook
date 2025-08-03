# BirdCLEF Audio Classification Notebook

This repository contains a notebook for bird sound classification using deep learning techniques and audio preprocessing pipelines. The project is built around the [BirdCLEF](https://www.kaggle.com/competitions/birdclef-2025) dataset and aims to classify 206 different bird species from audio recordings.

---

##  Project Highlights

-  **Audio Preprocessing**: 
  - Noise reduction using *noisereduce*. To improve signal quality, NoiseReduce was applied to remove any background and unwanted noise from segments of the audio file. 
  - To improve signal quality, *SileroVAD* was applied to identify and remove segments of audio containing human speech, which often overlapped with bird calls in the recordings. This helped isolate bird sounds by eliminating vocal noise from the dataset.
  - Conversion to Mel spectrograms

-  **Modeling**:
  - Feature extraction using pre-trained [YAMNet](https://tfhub.dev/google/yamnet/1).
    - *What is Transfer Learning?*
    Transfer learning is a machine learning technique in which knowledge gained through        one task or dataset is used to improve model performance on another related task and/or different dataset.
    Embeddings are extracted from a pre-trained model trained on large datasets to reuse learned data representations across different tasks. An embedding is a compact vector representation that captures essential features of data (in the context of this project- frequencies).
    - *Why Transfer Learning?*
      Classifying 206 distinct bird species based on rough audio recordings required huge amounts of classified data and computational power. To offset data deficiency and enhance computational efficiency and accuracy, Google's Yamnet model, a generalized sound classifier across various domains, such as Dogs, cats, construction noise, and Horn sounds, with a total of 512 classifications, was leveraged as a foundational feature extractor. 
  - Feedforward Neural Network with PyTorch
    - The EmbeddingClassifier is a neural network designed to classify audio embeddings from the YAMNet model into a fixed number of classes. It first applies adaptive average pooling to reduce variable-length inputs to a fixed 1024-dimensional vector. This is followed by two fully connected layers with ReLU activations and dropout for regularization, and a final layer that outputs class scores.
  - Class balancing via `class_weight`
      - Another major problem that was encountered was the imbalance in the data for each bird species. This imbalance in the data led to the underrepresentation of minority classes. Hence, to rectify this bias, class weights were used. Class weights are passed to the loss function; higher weights are assigned to the underrepresented classes and lower weights are assigned to the overrepresented ones. The loss function penalizes minority classes more than the majority ones. This led to an improvement in the bias and hence in the overall accuracy throughout all classes. 

-  **Evaluation**:
  - Train-test split used for validation
      - The model was trained on 5000 seconds (1486 files) of data. 
      - The model was tested on 3400 seconds (1155 files) of data.
      - Due to computational constraints, a subset of the data was used. This selection represented an optimal balance between maintaining model accuracy and staying within available resource limits.
  - AUC score was used to measure the model's performance. 

---

##  Requirements

Make sure the following packages are installed:

```bash
pip install pandas numpy matplotlib torch torchaudio noisereduce scikit-learn tensorflow_hub
```

---

##  How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Ri-Var/BirdCLEF-audio-classification-notebook.git
   cd BirdCLEF-audio-classification-notebook
   ```

2. Open `Classifier.ipynb` in Jupyter Notebook or VSCode.

3. Run all cells sequentially. Ensure internet access for downloading the YAMNet model and Silero VAD.

---

## Results

The model is evaluated on a train-test split using the following metrics:
- Area Under Curve (AUC): a metric used to evaluate the performance of binary or multi-class classification models, especially in imbalanced datasets. It measures a model's ability to rank positive instances higher than negative ones, across all classification thresholds. 
  The model after
  - removing Human Speech segments from the audio files using Silero VAD
  - using 3 randomly selected 1-sec clips from the audio files (fills in white noise if there aren't any available)
  - and using embeddings for Google's YAMNET audio classifier
  achieved an AUC score of **0.817**

---
## References

- [BirdCLEF Challenge](https://www.kaggle.com/competitions/birdclef-2025)
- [YAMNet (TensorFlow Hub)](https://tfhub.dev/google/yamnet/1)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Audio Deep Learning Made Simple: Sound Classification, Step-by-Step](https://medium.com/data-science/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
- [Transfer Learning](https://medium.com/@davidfagb/guide-to-transfer-learning-in-deep-learning-1f685db1fc94)
---

## Contact

**Ritish Varada**  
Email: [vrtish06@gmail.com](mailto:vrtish06@gmail.com)  
GitHub: [@Ri-Var](https://github.com/Ri-Var)
