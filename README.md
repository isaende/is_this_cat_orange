# is this cat orange??

A small Flask web app that receives a cat image, detects and crops the cat with YOLO, and estimates whether the cat has orange fur using OpenCV color analysis.

This project is open source and intended for educational/demonstration purposes.

## What It Uses

- **Flask** for the web server, upload form, and API routes.
- **Ultralytics YOLO** to detect the cat in the uploaded image.
- **OpenCV** to crop, segment, and analyze image colors.
- **NumPy** for pixel-level image processing.
- **HTML/CSS templates** for the simple browser interface.

## How It Works

1. The user uploads a cat image.
2. The backend validates the file type.
3. YOLO detects the cat and returns the best bounding box.
4. The app crops the cat from the original image.
5. OpenCV analyzes the crop using color heuristics.
6. The app returns a result such as `orange`, `orange and white`, `black`, `white`, `gray`, `tuxedo`, `brown/tabby`, or `mixed/unknown`.

## Requirements

Install the Python dependencies from `requirements.txt`:

```txt
flask
numpy
opencv-python
ultralytics
```

You also need a YOLO model file. By default, the app looks for:

- `yolo11x.pt` in the project root
- `models/yolo11x.pt`
- a custom path set with the `YOLO_MODEL_PATH` environment variable

## How To Run

1. Place the `yolo11x.pt` model in the project root or in `models/yolo11x.pt`.
   - Alternatively, set `YOLO_MODEL_PATH` to another model path.

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Start the server:

```powershell
python app.py
```

5. Open:

```txt
http://127.0.0.1:5000
```

## API Endpoint

`POST /api/analyze`

Send a `multipart/form-data` request with the field name `image`.

Example response:

```json
{
  "message": "yes, this cat is orange",
  "label": "orange",
  "is_orange": true,
  "orange_score": 0.43,
  "ratios": {
    "orange": 0.43,
    "black": 0.03,
    "white": 0.18,
    "gray": 0.05,
    "brown_tabby": 0.02
  },
  "cat_detection": {
    "x1": 120,
    "y1": 42,
    "x2": 640,
    "y2": 710,
    "confidence": 0.91
  },
  "original_url": "/static/uploads/original_cat_abc123.jpg",
  "crop_url": "/static/crops/cat_cat_abc123.jpg"
}
```

## Important Notes

- The fur-color classification is currently a color heuristic, not a second trained machine learning model.
- Results work best when the YOLO crop contains mostly the cat and little background.
- Lighting, shadows, filters, background colors, and partial cat crops can affect the result.
- The natural next step would be replacing the heuristic with a trained classifier for classes such as `orange`, `tuxedo`, `black`, `white`, `gray`, `calico`, and `tabby`.

## License Notice

This project uses Ultralytics YOLO, which is licensed under AGPL-3.0. This project is open source and intended for educational/demonstration purposes.
