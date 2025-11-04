# Energy Disaggregation using CNN (Seq2Point)

This project implements a **Convolutional Neural Network (CNN)** based **Seq2Point** model for **Non-Intrusive Load Monitoring (NILM)** — the task of estimating the power consumption of individual household appliances from aggregate mains readings.

The model learns to predict the power usage of a specific appliance at the midpoint of a sliding window of aggregate power data, allowing accurate disaggregation of energy consumption patterns.

---

## Project Overview

Energy disaggregation (or **NILM**) aims to decompose the total power signal of a home into individual appliance-level signals without using extra sensors.  
This project applies a **Seq2Point architecture**, inspired by the paper:

> Kelly, J., & Knottenbelt, W. (2015). *Neural NILM: Deep neural networks applied to energy disaggregation.*  
> Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments.

The network uses **1D Convolutional layers** to learn temporal patterns in energy consumption sequences.

---

## Model Architecture

The Seq2Point model is built using **Keras** with the following layers:

- `Conv1D` × 5 : Extract temporal features from aggregate signals  
- `Flatten` : Convert convolutional features into a dense vector  
- `Dense` layers : Predict the midpoint power value of the appliance  
- `Activation: ReLU` for non-linearity  
- `Loss: Mean Squared Error (MSE)`  
- `Optimizer: Adam`

The model takes a sliding window of mains readings (length = 599 samples) and predicts the corresponding midpoint target value.

---

## Dataset

The notebook uses data from the **REFIT dataset** (or any similar household energy dataset).  
Each dataset includes:
- `mains` readings (aggregate power consumption)
- `appliance` readings (ground truth)

Data is preprocessed by:
1. Normalizing power values (0–1 scaling)
2. Splitting data into **train** and **test sets**
3. Creating overlapping windows using a sliding window generator

---

## Requirements

Install the dependencies listed below before running the notebook:

```bash
pip install -r requirements.txt
```
##  How to Run

### Clone the repository:
```bash
git clone https://github.com/ManelHjaoujia/energy-disaggregation-cnn.git
cd energy-disaggregation-cnn
```
### Install dependencies:
```bash
pip install -r requirements.txt
```

### Open the notebook:
```bash
jupyter notebook Seq2point.ipynb
```
### Run all cells to:

- Load and preprocess data  
- Train the Seq2Point CNN model  
- Evaluate model performance (MAE, RMSE, visualizations)

---

## Results and Discussion

The Seq2Point CNN model was trained for **50 epochs**, showing a clear improvement in performance over time.  
The **training loss** and **mean absolute error (MAE)** decreased steadily, reaching around **0.10 loss** and **0.17 MAE** by epoch 29.  
However, the **validation loss and MAE plateaued around 0.42 and 0.33**, suggesting that the model began to overfit after approximately 15–20 epochs.

Overall, the model demonstrates good learning capability and convergence on the training set, but limited generalization on the validation set.  
This behavior is mainly due to:

- Restricted computational resources and early stopping based on validation stagnation  
- No extensive hyperparameter tuning (e.g., learning rate, batch size)  
- Possible dataset imbalance or insufficient variability  

Future experiments with **larger datasets, extended training, and stronger regularization** (e.g., dropout, batch normalization tuning) are expected to further improve validation performance and reduce overfitting.


---

##  Future Improvements

-  Increase epochs and use early stopping  
-  Implement GRU and LSTM models for comparison  
-  Add regularization and dropout layers  
-  Visualize learned filters and embeddings (t-SNE / PCA)  
-  Use larger datasets and additional appliances  
-  Integrate TensorBoard for monitoring training metrics  

---

## References

- Kelly, J., & Knottenbelt, W. (2015). *Neural NILM: Deep neural networks applied to energy disaggregation.*  
- Makonin, S. et al. (2016). *REFIT: Electrical Load Measurements (2013–2015).*  
- [Keras Documentation](https://keras.io)

---

## Author

**Manel Hjaoujia**  
Master’s student in Information Systems Engineering & Data Science
