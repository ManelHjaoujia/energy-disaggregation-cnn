# ⚡ Multi-Output Energy Disaggregation using Seq2Point CNN

This project implements a **multi-output Seq2Point Convolutional Neural Network (CNN)** to perform **energy disaggregation**, also known as **Non-Intrusive Load Monitoring (NILM)**.  
The goal is to estimate the individual power consumption of multiple household appliances using only the aggregate (total) power consumption signal.

---

## Project Overview

In smart energy systems, understanding how much electricity each appliance consumes helps with energy efficiency, anomaly detection, and demand forecasting.  
Traditional approaches require dedicated sensors for each appliance. This project instead uses **deep learning** to infer appliance-level data from aggregate consumption.

The proposed **Seq2Point CNN model** learns temporal patterns within a fixed-size time window of aggregate signals and predicts the corresponding appliance-level consumptions simultaneously.

---

## Model Architecture

The model is a **1D Convolutional Neural Network** built using TensorFlow/Keras with the following key layers:

- `Conv1D` layers (filters: 16 → 32 → 64 → 128)  
- `Flatten` layer  
- `Dense(1024)` fully connected layer with ReLU activation  
- `Dense(num_appliances)` output layer (multi-output regression)

Each output node corresponds to one household appliance.

**Model Summary:**

Total Parameters: 7,935,919                   
Trainable Parameters: 7,935,919                      
Non-trainable: 0                       


---

## Dataset

- **Source:** `all_buildings_data.csv`
- **Features:**  
  - `aggregate`: total power consumption of the building  
  - Multiple columns for individual appliance power readings
- **Preprocessing Steps:**  
  - Dropped unnecessary columns  
  - Standardized both input and target values using `StandardScaler`  
  - Created fixed-length sliding windows (`window_size = 60`)  

---

## Training Setup

- **Model:** Seq2Point CNN  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Metrics:** Mean Absolute Error (MAE)  
- **Epochs:** 50  
- **Batch Size:** 32  
- **Validation Split:** 20%  

**Training Environment:**  
Limited computational resources were available, so the model was trained for 50 epochs.  
Although results are promising, extended training and hyperparameter tuning are expected to further improve performance.

---

## Results

| Metric | Score |
|:--------|:------|
| **Mean Absolute Error (MAE)** | 0.3386 |
| **Mean Squared Error (MSE)** | 0.4859 |
| **R² Score** | 0.5245 |

The model successfully learned to approximate appliance-level power usage patterns with moderate accuracy given the constraints.

---

## Training Curves

**Loss Over Epochs**
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
```

**MAE Over Epochs**
```python
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
```
Both training and validation curves show stable convergence, indicating a well-generalized model.             

---

## Example Prediction

An example inference on a 60-point aggregate window shows the predicted energy usage per appliance:

| Appliance         | Predicted Consumption (kWh) |
|-------------------|-----------------------------|
| Laptop Computer   | 0.00                        |
| Television        | 5.48                        |
| HTPC              | 0.68                        |
| Microwave         | 0.49                        |
| Audio Amplifier   | 1.42                        |
| Immersion Heater  | 0.57                        |
| Fridge            | 3.33                        |
| ...               | ...                         |

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Future Improvements

- Extend training epochs and fine-tune hyperparameters  
- Implement dropout and batch normalization for regularization  
- Integrate TensorBoard for live training monitoring  
- Add visualization of appliance embeddings (t-SNE or PCA)  
- Explore hybrid models (CNN-LSTM / Transformer-based NILM)  

---

## Author

**Manel Hjaoujia**  
Master’s Student in Information Systems Engineering & Data Science
