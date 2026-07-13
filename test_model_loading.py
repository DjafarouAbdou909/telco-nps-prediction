import joblib
import json
from pathlib import Path

MODELS_DIR = Path("models")  

def test_model(filename, expected_type_name):
    path = MODELS_DIR / filename
    print(f"--- {filename} ---")
    if not path.exists():
        print(f"  [ERREUR] Fichier introuvable : {path.resolve()}")
        return
    try:
        obj = joblib.load(path)
        print(f"  [OK] Chargé avec succès — type : {type(obj)}")
        if hasattr(obj, "predict_proba"):
            print(f"  [OK] predict_proba disponible")
        elif isinstance(obj, dict) and "model" in obj:
            print(f"  [OK] dict avec clé 'model' — type interne : {type(obj['model'])}")
    except Exception as e:
        print(f"  [ERREUR] Erreur au chargement : {e}")
    print()


if __name__ == "__main__":
    test_model("nps_model_logreg.joblib", "Pipeline")
    test_model("nps_model_gradientboosting.joblib", "HistGradientBoostingClassifier")
    test_model("nps_model_ordinal.joblib", "dict (preprocessor + model)")

    meta_path = MODELS_DIR / "model_metadata.json"
    print("--- model_metadata.json ---")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  [OK] Lu avec succès")
        print(f"  Modèle principal : {meta.get('primary_model')}")
        print(f"  Raison du choix  : {meta.get('primary_model_reason')}")
    else:
        print(f"  [ERREUR] Fichier introuvable : {meta_path.resolve()}")