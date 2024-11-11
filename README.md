# Pneumonia-Detection
Here's a README file for your project:

---

# Pneumonia Detection from Chest X-Ray

This project uses a Convolutional Neural Network (CNN) based on ResNet50 to classify chest X-ray images as either "Normal" or "Pneumonia". The model is trained and fine-tuned using TensorFlow and Keras and deployed with a user-friendly interface using Streamlit.

## Project Structure

- **Training and Evaluation**: The project involves using a pre-trained ResNet50 model for feature extraction, with additional custom layers for binary classification.
- **Image Data Generators**: For data augmentation and preprocessing.
- **Streamlit Deployment**: A Streamlit app is provided for users to upload their X-ray images and receive predictions.

## Dataset

The chest X-ray images dataset used here contains images for both training and validation, with labels for "Normal" and "Pneumonia" categories.

- **Training Data Directory**: `/content/drive/MyDrive/archive (14)/chest_xray/train`
- **Validation Data Directory**: `/content/drive/MyDrive/archive (14)/chest_xray/val`

## Requirements

Install the required libraries:
```bash
pip install tensorflow numpy opencv-python-headless matplotlib streamlit pillow
```

## Training Process

1. **Data Augmentation**: Images are rescaled, sheared, zoomed, and flipped horizontally to improve model robustness.
2. **Model Architecture**: A ResNet50 model (pre-trained on ImageNet) is used as a base, followed by global pooling and fully connected layers for binary classification.
3. **Training**: The model is trained for 10 epochs, using binary cross-entropy loss and Adam optimizer.
4. **Evaluation**: After training, the model's performance is evaluated on the validation dataset, and results are displayed.

## Streamlit App for Inference

The Streamlit app enables users to upload chest X-ray images to check for signs of pneumonia.

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image**: Upload a chest X-ray image in `.jpg`, `.jpeg`, or `.png` format.

3. **Prediction**: The model will predict the likelihood of the uploaded image indicating pneumonia or being normal, with a confidence score displayed.

## Code Description

### Data Augmentation and Preprocessing

```python
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)
```

### Model Architecture

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```

### Streamlit Inference

1. **Load Trained Model**: The model (`pneumonia_detection_model.keras`) is loaded.
2. **Image Preprocessing**: Uploaded images are resized and normalized.
3. **Prediction**: The model predicts the probability of pneumonia, displayed with a confidence score.

```python
model = tf.keras.models.load_model('path_to_model/pneumonia_detection_model.keras')
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])
```

## Results

Validation accuracy and loss metrics are visualized, along with predictions for each uploaded X-ray image.

## Files

- `ACVCA.ipynb`: Notebook containing training code.
- `app.py`: Streamlit app for real-time pneumonia detection.

## References

- Chest X-ray dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

