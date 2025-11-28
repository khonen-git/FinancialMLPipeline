# Guide de Test - Financial ML Pipeline

## Installation

### 1. Installer les dépendances

```bash
# Avec conda (recommandé)
conda env create -f environment.yml
conda activate trading-ml

# Ou avec pip
pip install -r requirements.txt
pip install -e .
```

### 2. Vérifier l'installation

```bash
python -c "import pandas, numpy, sklearn, mlflow; print('✅ Imports OK')"
```

## Préparer les Données

### 1. Créer un échantillon pour test rapide

```bash
# Créer un sample de 50k lignes depuis EURUSD 2023
python scripts/prepare_data.py --create-sample --asset EURUSD --year 2023 --n-rows 50000
```

### 2. Convertir les fichiers complets (optionnel)

```bash
# Convertir EURUSD 2023 (training)
python scripts/prepare_data.py --convert data/eurusd-tick-2023-01-01-2024-01-01.csv

# Convertir EURUSD 2024 (testing)
python scripts/prepare_data.py --convert data/eurusd-tick-2024-01-01-2025-01-01.csv

# Ou convertir tous les fichiers CSV
python scripts/prepare_data.py --convert-all
```

### 3. Vérifier les données

```bash
# Inspecter le sample
python scripts/inspect_data.py data/raw/EURUSD_2023_sample.parquet
```

## Lancer les Tests Unitaires

### 1. Tests du Session Calendar

```bash
python -m pytest tests/test_session_calendar.py -v
```

**Tests inclus** (15 tests) :
- ✅ Détection des weekends (Saturday, Sunday, weekdays)
- ✅ Calcul de session_end (regular days, Friday)
- ✅ Détection "near session end"
- ✅ Filtrage des ticks par session
- ✅ Calcul du temps jusqu'à session_end
- ✅ Edge cases (midnight, weekend trading enabled)

### 2. Tests du Triple Barrier

```bash
python -m pytest tests/test_triple_barrier.py -v
```

**Tests inclus** (10 tests) :
- ✅ Hit TP (Take Profit)
- ✅ Hit SL (Stop Loss)
- ✅ Hit Time Barrier
- ✅ Horizon effectif (regular, near session end, past session)
- ✅ Skip events trop proches de session_end
- ✅ Entry @ ask, Exit @ bid (validation des prix)

### 3. Tests de Construction de Bars

```bash
python -m pytest tests/test_bars.py -v
```

**Tests inclus** (10 tests) :
- ✅ Tick bars (every N ticks)
- ✅ Volume bars
- ✅ Dollar bars
- ✅ Logique OHLC (High >= Low, etc.)
- ✅ Métadonnées (tick_count, spread_mean, spread_std)
- ✅ Edge cases (empty, few ticks, no volume data)

### 4. Tests de Détection de Schéma

```bash
python -m pytest tests/test_schema_detection.py -v
```

**Tests inclus** (8 tests) :
- ✅ Schéma valide (Dukascopy format)
- ✅ Colonnes manquantes
- ✅ Timestamps invalides
- ✅ Prix négatifs
- ✅ Spreads négatifs (bid > ask)
- ✅ Edge cases (empty, extra columns)

### 5. Lancer tous les tests

```bash
# Tous les tests avec rapport détaillé
python -m pytest tests/ -v

# Avec coverage
python -m pytest tests/ --cov=src --cov-report=html

# Tests spécifiques
python -m pytest tests/test_triple_barrier.py::TestTripleBarrierLabeler::test_label_single_event_tp_hit -v
```

## Test End-to-End

### 1. Créer une configuration de test

Créer `configs/experiment/test_sample.yaml` :

```yaml
# @package _global_
defaults:
  - override /assets: EURUSD
  - override /data/bars: base_bars
  - override /session: base_session
  - override /labeling: triple_barrier
  - override /features: base_features
  - override /models: rf_cpu
  - override /validation: tscv
  - override /backtest: base_bt
  - override /risk: base_risk
  - override /reporting: base_report

experiment:
  name: "test_sample_eurusd"
  description: "Quick test with EURUSD sample"
  tags: ["test", "sample", "eurusd"]

data:
  dukascopy:
    raw_dir: "data/raw"
    filename: "EURUSD_2023_sample.parquet"  # Use sample
  
  bars:
    type: "tick"
    threshold: 500  # Small threshold for quick test

labeling:
  triple_barrier:
    tp_ticks: 50  # Smaller for quick hits
    sl_ticks: 50
    max_horizon_bars: 20  # Shorter horizon
    min_horizon_bars: 5

validation:
  n_splits: 2  # Just 2 folds for quick test
  train_duration: 5000
  test_duration: 1000

models:
  random_forest:
    params:
      n_estimators: 50  # Fewer trees for speed
      max_depth: 5

risk:
  mc_simulations: 1000  # Fewer simulations for speed
```

### 2. Valider la configuration

```bash
python scripts/validate_config.py experiment=test_sample
```

### 3. Lancer l'expérience de test

```bash
python run_experiment.py experiment=test_sample
```

### 4. Monitorer avec MLflow

```bash
# Dans un autre terminal
mlflow ui

# Ouvrir http://localhost:5000
```

## Test avec Données Complètes

### 1. Configuration pour training/backtest

Créer `configs/experiment/eurusd_2023_2024.yaml` :

```yaml
# @package _global_
defaults:
  - override /assets: EURUSD
  - override /data/bars: base_bars
  - override /session: base_session
  - override /labeling: triple_barrier
  - override /features: base_features
  - override /models: rf_cpu
  - override /validation: tscv
  - override /backtest: base_bt
  - override /risk: base_risk
  - override /reporting: base_report

experiment:
  name: "eurusd_2023_train_2024_test"
  description: "Train on 2023, test on 2024"
  tags: ["eurusd", "2023", "2024", "full"]

data:
  dukascopy:
    raw_dir: "data/raw"
    filename: "EURUSD_2023.parquet"  # Full 2023 data for training
  
  test_data:
    filename: "EURUSD_2024.parquet"  # Full 2024 for backtest

# Use default parameters for production
```

### 2. Lancer l'expérience complète

```bash
python run_experiment.py experiment=eurusd_2023_2024
```

## Résultats Attendus

### Tests Unitaires

✅ **43 tests au total** :
- 15 tests Session Calendar
- 10 tests Triple Barrier  
- 10 tests Bar Construction
- 8 tests Schema Detection

### Test End-to-End (Sample)

L'expérience de test devrait :
1. ✅ Charger le sample (50k ticks)
2. ✅ Construire ~100 bars (tick bars, threshold=500)
3. ✅ Créer ~30-50 features
4. ✅ Générer ~50-80 labels (triple barrier)
5. ✅ Entraîner les HMMs (macro + micro)
6. ✅ Entraîner le Random Forest
7. ✅ Effectuer 2 folds de CV
8. ✅ Générer un rapport HTML
9. ✅ Logger dans MLflow

**Durée estimée** : 2-5 minutes

### Test Complet (2023 + 2024)

Avec les données complètes :
- 🕐 Durée estimée : 30-60 minutes
- 📊 Millions de ticks traités
- 🎯 Milliers de labels générés
- 📈 Backtest réaliste sur 2024

## Troubleshooting

### Erreur : Module not found

```bash
# Réinstaller le package
pip install -e .
```

### Erreur : File not found

```bash
# Vérifier que les fichiers Parquet existent
ls -lh data/raw/

# Recréer le sample si nécessaire
python scripts/prepare_data.py --create-sample --asset EURUSD --year 2023 --n-rows 50000
```

### Erreur : Out of memory

```bash
# Réduire la taille du sample
python scripts/prepare_data.py --create-sample --asset EURUSD --year 2023 --n-rows 10000

# Ou ajuster les paramètres de configuration (moins de bars, features, etc.)
```

### Tests qui échouent

```bash
# Lancer un test spécifique en mode verbose
python -m pytest tests/test_triple_barrier.py::TestTripleBarrierLabeler::test_label_single_event_tp_hit -vv

# Afficher les prints
python -m pytest tests/test_session_calendar.py -v -s
```

## Validation Finale

### Checklist avant production

- [ ] Tous les tests unitaires passent (43/43)
- [ ] Test end-to-end avec sample réussi
- [ ] MLflow tracking fonctionne
- [ ] Rapport HTML généré
- [ ] Test avec données complètes 2023
- [ ] Backtest sur 2024 cohérent
- [ ] Métriques dans les ranges attendus :
  - Win rate : 40-60%
  - Sharpe ratio : > 0.5
  - Max drawdown : < 20%
  - Probability of ruin : < 10%

## Prochaines Étapes

1. **Optimisation des hyperparamètres**
   - Grid search sur TP/SL ticks
   - Tuning des modèles (RF, HMM)

2. **Ajout de features**
   - Fractional differencing
   - Volatility regimes
   - Order flow features avancées

3. **Tests sur d'autres assets**
   - GBPUSD, USDJPY, etc.
   - Multi-asset strategy

4. **Production**
   - CI/CD avec GitHub Actions
   - Docker deployment
   - Live trading integration

---

*Pour toute question, consulter la documentation dans `docs/` ou `QUICKSTART.md`*

