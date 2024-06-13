from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import shutil

app = FastAPI()

# Load the model
model = load_model('E:/Image recognition/model/celeb_model3.h5')

# Define class names
class_names = [
    'Brad Pitt', 'Chris Hemsworth', 'Christiano Ronaldo', 'Elon Musk', 'Eminem',
    'Garry Kaspov', 'Jeff Bezoz', 'Leonardo di Caprio', 'Lionel Messi',
    'Magnus Carlsen', 'Mark Zuckerburg', 'Mike Tyson', 'Muhammad Ali',
    'Novak Djokovick', 'Sharukh Khan', 'Tom Holland', 'Will Smith'
]

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Save the uploaded file
    file_location = f"static/uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    # Process the image and get predictions
    img = image.load_img(file_location, target_size=(225, 225))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Normalize image
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    if confidence < 0.70:
        predicted_class = "Unknown"
    else:
        predicted_class = class_names[predicted_class_index]
    
    response = {
        "predicted_class": predicted_class,
        "confidence": f"{confidence * 100:.2f}%"  
    }

    return JSONResponse(content=response)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
