# is this cat orange??

Backend Flask para receber uma imagem, detectar/cropar o gato com YOLO e estimar a cor da pelagem com OpenCV.

## Como rodar

1. Coloque o modelo `yolo11x.pt` na raiz do projeto ou em `models/yolo11x.pt`.
   - Alternativa: defina `YOLO_MODEL_PATH` apontando para outro caminho.
2. Instale as dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Inicie o servidor:

```powershell
python app.py
```

4. Abra `http://127.0.0.1:5000`.

## Endpoint para o futuro frontend

`POST /api/analyze`

Envie multipart/form-data com o campo `image`.

Exemplo de resposta:

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

## Observacao

A classificacao de pelagem ainda e uma heuristica de cor, nao um segundo modelo treinado. Ela funciona melhor quando o crop do gato tem pouca interferencia do fundo. O proximo passo natural e trocar essa heuristica por um classificador treinado com classes como `orange`, `tuxedo`, `black`, `white`, `gray`, `calico` e `tabby`.
