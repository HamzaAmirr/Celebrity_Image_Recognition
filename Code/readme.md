# celebrity_Image_Recognition.ipynb contains the code for all of the following

## 1. Data Pre-Processing

- The images were loaded as `tf.data.Datasets` and divided into training and validation subsets using the TensorFlow `tf.keras.preprocessing.image_dataset_from_directory` library.
- The dataset was split into 80% training and 20% validation sets.
- Data augmentation was performed using TensorFlow data augmentation layers to enhance the model's ability to generalize.

## 2. Model Architecture

- Transfer learning was employed due to the complexity of the problem and the small dataset size.
- EfficientNetB0 was chosen as the base model due to its transfer learning capability, performance, scalability, regularization techniques, and model size.
- The initial architecture involved removing the top layer of the EfficientNetB0 model, inserting a global average pooling layer, and adding two densely connected neural network layers with dropout.
- Overfitting occurred with the initial architecture, leading to the exploration of different model configurations.
- Model iterations ('model1', 'model2', 'model3') were evaluated, adjusting the number of trainable layers in the EfficientNetB0 model.
- The optimizer, loss function, and metrics used during training were Adam, sparse categorical cross-entropy, and accuracy, respectively.
- Callbacks were utilized for learning rate scheduling, model checkpointing, and early stopping to optimize model training.

## 3. Model Evaluation

- Each model iteration was evaluated using loss and accuracy graphs for both training and validation datasets.
- 'Model3' was selected as the final model based on its performance metrics.
- Model evaluation included standard metrics such as accuracy, precision, recall, and F1-score, as well as a confusion matrix to assess classification performance.

## 4. Model Saving and Predictions

- The final model ('model3') was saved after training.
- A function was created to preprocess single images for model input.
- The model was tested on new images, with classifications below a confidence threshold labeled as "Unknown" and high-confidence misclassifications noted.

#### For further details, please refer to the project code and documentation section 3.2, 3.3, 3.4, 3.5.
