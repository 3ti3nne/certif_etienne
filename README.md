# Cibler une campagne bancaire de manière responsable

Certification Dev-id sur le jeu UCI Bank Marketing. Le livrable est le notebook `notebooks/etienne_roubaud_certif_v1.ipynb`. Le journal de bord est dans `notebooks/journal-de-bord.ipynb`.

## Lancer le projet

```
python -m venv .venv
.venv\Scripts\activate        # Linux et macOS : source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m scripts.train
docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api
```

`python -m scripts.train` doit tourner avant `docker build` : il crée le dossier `models/`, absent du dépôt, que l'image Docker copie.

Réentraînement : `python -m scripts.retrain`. Interface : `streamlit run streamlit_app.py`. Suivi des modèles : `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
