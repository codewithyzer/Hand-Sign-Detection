## 1. Install Python 3.10

🔗 Download Python 3.10 from the official page:  
[https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)

Make sure to:

- Check `Add Python to PATH` during installation

## 2. Navigate to your project folder

- cd into project folder
  cd path/to/Hand-Sign-Detection

- Create a virtual environment named env
  `py -3.10 -m venv env`

- Activate virtual environment (you should see "(env)" at the start if succesfully activated)
  `env\Scripts\Activate`

- select you created env folder as your interpreter (`OPTIONAL`: if env didn't get automatically selected)

## 3. Install Project Dependencies

- in your vscode terminal
  `pip install -r requirements.txt`

## 4. Run the data collection program

- in data_collection.py change "save_folder" variable depending on the letter you want collect images from
  `save_folder = "data/<letter>"` make sure letter is on uppercase like `save_folder = "data/A"`

- run data_collection program in vscode terminal
  `python data_collection.py`

- capture picture
  `hold "s" key` to capture on your keyboard and save your images

- exit the program
  `press "q" key`

## 5. Train a model

- go to `https://teachablemachine.withgoogle.com/train`

  - click image project then standard image model

  - change the class name based on the images you want to upload (`EXAMPLE`: image data set letter A `change class name to A`)

  - upload images by clicking `upload` then go porjects directory `Hand Sign Detection\data\<letter>\` then select all the image for that letter

  - when done click `Train model`

  - click `export model`

  - choose `tensorflow` in the header then select `keras` then download model after beging converted

## 6. Import model

- extract the downloaded model
  find `keras.model.h5` and `labels.txt` in the extracted file
  move `keras.model.h5` and `labels.txt` into `Hand Sign Detection\model`

## 5. Run the application model

- In your vscode terminal
  python test.py
