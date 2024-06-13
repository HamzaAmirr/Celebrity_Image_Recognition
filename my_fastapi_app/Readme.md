## To directly run this web application, follow these steps:

1. **Clone the Repository**:
   - Clone the repository to your local machine using the following command:
     ```
     git clone <repository-url>
     ```

2. **Install Dependencies**:
   - Install the necessary dependencies listed in the references section in the Documentation.docx file.

3. **Download the Trained Model**:
   - Download the trained model from the 'model' folder in the repository.
   - Update its path in the 'model' variable in the 'main' file.

4. **Start the FastAPI Server**:
   - Start the FastAPI server with uvicorn using the following command:
     ```
     uvicorn main:app --reload
     ```

5. **Access the Web Application**:
   - Open your web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
   
6. **Upload Images**:
   - Upload images of celebrities to see the prediction results
