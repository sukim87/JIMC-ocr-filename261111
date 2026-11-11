V.261111 Streamlit Community Cloud deployment package

Files to upload to the GitHub repository root:
- app.py
- requirements.txt
- run.py (not used by Community Cloud, retained for local/EXE execution)

Important deployment settings:
- Repository: sukim87/jimc-ocr-filename261111
- Branch: main
- Main file path: app.py
- Python version: 3.12

Deployment order:
1. Replace all three files in the GitHub repository root.
2. Delete the existing Streamlit Community Cloud app only. Do not delete the GitHub repository.
3. Create the app again.
4. In Advanced settings select Python 3.12.
5. Deploy with main file path app.py.

This package includes:
- Lazy EasyOCR loading after clicking Analyze
- CPU-only PyTorch dependencies
- OCR result/session preservation
- Reduced repeated OCR
- Numeric image preprocessing
