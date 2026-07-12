from __future__ import annotations

import os
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.serving import WSGIRequestHandler
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = FRONTEND_DIR / "uploads"
CROP_DIR = FRONTEND_DIR / "crops"
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
HAPPY_IMAGE = "images/cat_happy.png"
CONFUSED_IMAGE = "images/confused.jpg"
SILLY_IMAGE = "images/silly.png"
NO_CAT_MESSAGE = "that's not a cat"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    template_folder=str(FRONTEND_DIR / "html"),
)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

_yolo_model: Any | None = None


class QuietRequestHandler(WSGIRequestHandler):
    def log(self, type: str, message: str, *args: Any) -> None:
        return


logging.getLogger("werkzeug").disabled = True


@app.after_request
def add_api_cors_headers(response: Any) -> Any:
    if request.path.startswith("/api/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

    return response


def api_json(payload: dict[str, Any], status_code: int) -> tuple[Any, int]:
    response = jsonify(payload)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response, status_code


def reaction_image_for_result(is_orange: bool, label: str) -> str:
    if is_orange:
        return f"/frontend/{HAPPY_IMAGE}"

    if label == "mixed/unknown":
        return f"/frontend/{CONFUSED_IMAGE}"

    return f"/frontend/{SILLY_IMAGE}"


def reaction_image_for_error(message: str) -> str:
    if message == NO_CAT_MESSAGE:
        return f"/frontend/{SILLY_IMAGE}"

    return f"/frontend/{SILLY_IMAGE}"


def resolve_model_path() -> Path:
    env_path = os.getenv("YOLO_MODEL_PATH")

    if env_path:
        path = Path(env_path).expanduser()
        return path if path.is_absolute() else BASE_DIR / path

    for candidate in (BASE_DIR / "yolo11x.pt", BASE_DIR / "models" / "yolo11x.pt"):
        if candidate.exists():
            return candidate

    return BASE_DIR / "yolo11x.pt"


MODEL_PATH = resolve_model_path()


@dataclass
class CatBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class ColorAnalysis:
    label: str
    message: str
    is_orange: bool
    orange_score: float
    ratios: dict[str, float]


class AnalysisError(Exception):
    status_code = 400


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return suffix in ALLOWED_EXTENSIONS


def load_yolo_model() -> Any:
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    if not MODEL_PATH.exists():
        raise AnalysisError(
            f"Modelo YOLO nao encontrado em '{MODEL_PATH}'. "
            "Coloque o yolo11x.pt na raiz do projeto ou defina YOLO_MODEL_PATH."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise AnalysisError(
            "A dependencia 'ultralytics' nao esta instalada. "
            "Rode: pip install -r requirements.txt"
        ) from exc

    _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model


def image_from_upload(upload: FileStorage) -> np.ndarray:
    if not upload or not upload.filename:
        raise AnalysisError("Envie uma imagem para analisar.")

    if not allowed_file(upload.filename):
        raise AnalysisError("Formato invalido. Use JPG, JPEG, PNG ou WEBP.")

    raw_bytes = np.frombuffer(upload.read(), np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise AnalysisError("Nao consegui abrir essa imagem. Tente outro arquivo.")

    return image


def save_image(image: np.ndarray, directory: Path, original_name: str, prefix: str) -> str:
    safe_name = secure_filename(original_name) or "image.jpg"
    stem = Path(safe_name).stem[:50]
    filename = f"{prefix}_{stem}_{uuid4().hex[:10]}.jpg"
    output_path = directory / filename
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return filename


def detect_cat(image: np.ndarray) -> CatBox:
    model = load_yolo_model()
    results = model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    if not results:
        raise AnalysisError(NO_CAT_MESSAGE)

    result = results[0]
    boxes = getattr(result, "boxes", None)

    if boxes is None or len(boxes) == 0:
        raise AnalysisError(NO_CAT_MESSAGE)

    names = getattr(result, "names", None) or getattr(model, "names", {})
    name_items = names.items() if isinstance(names, dict) else enumerate(names)
    class_names = {int(key): str(value).lower() for key, value in name_items}
    cat_candidates: list[tuple[float, CatBox]] = []

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, str(class_id)).lower()

        if class_name != "cat" and len(class_names) != 1:
            continue

        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().tolist()
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        confidence = float(box.conf[0])
        score = width * height * confidence

        cat_candidates.append(
            (
                score,
                CatBox(
                    x1=max(0, int(round(x1))),
                    y1=max(0, int(round(y1))),
                    x2=min(image.shape[1], int(round(x2))),
                    y2=min(image.shape[0], int(round(y2))),
                    confidence=confidence,
                ),
            )
        )

    if not cat_candidates:
        raise AnalysisError(NO_CAT_MESSAGE)

    return max(cat_candidates, key=lambda item: item[0])[1]


def crop_cat(image: np.ndarray, box: CatBox, padding_ratio: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    box_width = box.x2 - box.x1
    box_height = box.y2 - box.y1
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    x1 = max(0, box.x1 - pad_x)
    y1 = max(0, box.y1 - pad_y)
    x2 = min(width, box.x2 + pad_x)
    y2 = min(height, box.y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        raise AnalysisError("O crop do gato ficou invalido. Tente outra imagem.")

    return image[y1:y2, x1:x2].copy()


def central_ellipse_mask(height: int, width: int) -> np.ndarray:
    y_grid, x_grid = np.ogrid[:height, :width]
    center_x = width / 2
    center_y = height / 2
    radius_x = max(width * 0.46, 1)
    radius_y = max(height * 0.46, 1)
    mask = ((x_grid - center_x) ** 2 / radius_x**2) + (
        (y_grid - center_y) ** 2 / radius_y**2
    )
    return mask <= 1


def foreground_pixels(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]

    if height < 30 or width < 30:
        return crop.reshape(-1, 3)

    grabcut_mask = np.zeros((height, width), np.uint8)
    rect_margin_x = max(1, int(width * 0.04))
    rect_margin_y = max(1, int(height * 0.04))
    rect = (
        rect_margin_x,
        rect_margin_y,
        max(1, width - rect_margin_x * 2),
        max(1, height - rect_margin_y * 2),
    )

    try:
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            crop,
            grabcut_mask,
            rect,
            background,
            foreground,
            4,
            cv2.GC_INIT_WITH_RECT,
        )
        mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(bool)

        foreground_ratio = float(mask.mean())
        if 0.06 <= foreground_ratio <= 0.96:
            return crop[mask]
    except cv2.error:
        pass

    return crop[central_ellipse_mask(height, width)]


def ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return round(float(mask.mean()), 3)


def classify_fur_color(crop: np.ndarray) -> ColorAnalysis:
    pixels = foreground_pixels(crop)

    if pixels.size == 0:
        raise AnalysisError("Nao sobrou area suficiente do gato para analisar a cor.")

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    blue = pixels[:, 0].astype(np.float32)
    green = pixels[:, 1].astype(np.float32)
    red = pixels[:, 2].astype(np.float32)

    orange = (
        (hue >= 3)
        & (hue <= 24)
        & (saturation >= 65)
        & (value >= 70)
        & (red >= green * 1.22)
        & (red >= blue * 1.75)
    )
    cream_or_light_orange = (
        (hue >= 5)
        & (hue <= 25)
        & (saturation >= 45)
        & (saturation < 145)
        & (value >= 135)
        & (red >= green * 1.16)
        & (red >= blue * 1.55)
        & (green >= blue * 1.12)
    )
    black = value <= 58
    white = (saturation <= 38) & (value >= 188)
    gray = (saturation <= 45) & (value > 58) & (value < 188)
    orange_family = orange | cream_or_light_orange
    brown_tabby = (
        (hue <= 25)
        & (saturation >= 35)
        & (value > 45)
        & (value < 165)
        & (red >= blue * 1.10)
        & ~orange_family
    )

    ratios = {
        "orange": ratio(orange_family),
        "black": ratio(black),
        "white": ratio(white),
        "gray": ratio(gray),
        "brown_tabby": ratio(brown_tabby),
    }

    orange_score = ratios["orange"]
    brown_tabby_dominates = (
        ratios["brown_tabby"] >= 0.30
        and ratios["brown_tabby"] >= orange_score * 1.35
    )
    is_orange = orange_score >= 0.16 and not brown_tabby_dominates

    if is_orange and ratios["black"] >= 0.16 and ratios["white"] >= 0.12:
        label = "calico/tortie"
        message = "yes, this cat has orange fur (looks calico or tortie)"
    elif is_orange and ratios["white"] >= 0.18:
        label = "orange and white"
        message = "yes, this cat is orange and white"
    elif is_orange:
        label = "orange"
        message = "yes, this cat is orange"
    elif ratios["black"] >= 0.23 and ratios["white"] >= 0.18 and (
        ratios["black"] + ratios["white"]
    ) >= 0.55:
        label = "tuxedo"
        message = "no, that's a tuxedo cat"
    elif ratios["black"] >= 0.48:
        label = "black"
        message = "no, that's a black cat"
    elif ratios["white"] >= 0.50:
        label = "white"
        message = "no, that's a white cat"
    elif ratios["gray"] >= 0.36:
        label = "gray"
        message = "no, that's a gray cat"
    elif ratios["brown_tabby"] >= 0.18:
        label = "brown/tabby"
        message = "no, that looks like a brown tabby cat"
    else:
        label = "mixed/unknown"
        message = "no, I can't confidently call this cat orange"

    return ColorAnalysis(
        label=label,
        message=message,
        is_orange=is_orange,
        orange_score=orange_score,
        ratios=ratios,
    )


def analyze_image(upload: FileStorage) -> dict[str, Any]:
    ensure_runtime_dirs()
    original_image = image_from_upload(upload)
    original_filename = upload.filename or "cat.jpg"
    original_saved = save_image(original_image, UPLOAD_DIR, original_filename, "original")

    cat_box = detect_cat(original_image)
    cropped_cat = crop_cat(original_image, cat_box)
    crop_saved = save_image(cropped_cat, CROP_DIR, original_filename, "cat")
    color = classify_fur_color(cropped_cat)

    return {
        "message": color.message,
        "label": color.label,
        "is_orange": color.is_orange,
        "orange_score": color.orange_score,
        "ratios": color.ratios,
        "cat_detection": asdict(cat_box),
        "original_url": url_for("static", filename=f"uploads/{original_saved}"),
        "crop_url": url_for("static", filename=f"crops/{crop_saved}"),
        "reaction_image_url": reaction_image_for_result(color.is_orange, color.label),
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/analyze")
def analyze_page() -> tuple[str, int] | str:
    try:
        result = analyze_image(request.files.get("image"))  # type: ignore[arg-type]
        return render_template("result.html", result=result)
    except AnalysisError as exc:
        return render_template("index.html", error=str(exc)), exc.status_code


@app.post("/api/analyze")
def analyze_api() -> tuple[Any, int]:
    try:
        result = analyze_image(request.files.get("image"))  # type: ignore[arg-type]
        return api_json(result, 200)
    except AnalysisError as exc:
        message = str(exc)
        return api_json(
            {"error": message, "reaction_image_url": reaction_image_for_error(message)},
            exc.status_code,
        )
    except RequestEntityTooLarge:
        message = "image too large. Use an image up to 12 MB."
        return api_json(
            {"error": message, "reaction_image_url": reaction_image_for_error(message)},
            413,
        )
    except Exception:
        message = "Could not analyze the image at this time."
        return api_json(
            {"error": message, "reaction_image_url": reaction_image_for_error(message)},
            500,
        )


if __name__ == "__main__":
    ensure_runtime_dirs()
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() not in {"0", "false", "no"}
    app.run(
        host="127.0.0.1",
        port=port,
        debug=debug,
        use_reloader=debug,
        request_handler=QuietRequestHandler,
    )
