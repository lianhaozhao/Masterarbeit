
# Masterarbeit – Unsupervised Domain Adaptation zur Verschleißüberwachung in Stanzprozessen

Dies ist der Code-Teil meiner Masterarbeit. In dieser Arbeit wird untersucht, wie unüberwachte Domain-Adaptation-Methoden genutzt werden können, um ungelabelte Datensätze für die Modellbildung einzusetzen. Ziel ist es, Methoden zu erforschen und zu implementieren, die die Generierung von Pseudo-Labels ermöglichen, um die Datensätze mit künstlichen Zielgrößen für überwachtes Lernen nutzbar zu machen.

---

## 📁 Projektstruktur

```
.
├── baseline/
│   ├── model/             # Trainings- und Testskripte für Modelle (.py und .ipynb)
│
├── configs/
│   └── default.yaml             # Konfigurationsdatei für Hyperparameter
│
├── datasets/                    # Datensatzverzeichnis (manuell hinzufügen)
│
├── method_pseudo/               # method_pseudo
│   └── m_pseudo
│ 
├── method_DANN/                # method_DANN
│   └── m_DANN
│ 
├── models/                      # Modellarchitekturen (z.B. Flexible_CNN)
│
├── preprocessing/               # Datenvorverarbeitung
│
├── utils/
│   ├── train_utils.py           # Hilfsfunktionen für das Training
│
├── data_preprocessing.py       # Skript zur Erstellung von Datenpfad-Textdateien
├── outliers.py                 # Ausreißerbehandlung
├── requirements.txt            # Abhängigkeiten für Python
├── Denken und Reflexion.md     # Denken und Reflexion
└── README.md                   # Diese Dokumentation
```

---

## Schnellstart

### 1. Abhängigkeiten installieren

Virtuelle Umgebung empfohlen:

```bash
pip install -r requirements.txt
```

---

### 2. Daten vorbereiten

Nutze `data_preprocessing.py`, um Datenpfade aus `datasets` nach `datasets/source/` zu generieren:

- Trainingsdaten: `train/DC_T197_RP.txt`
- Validierungsdaten: `validation/HC_T197_RP.txt`
- Testdaten: `test/DC_T197_RP.txt` (für `baseline_test`)

---

### 3. Modelltraining

Ausführen:

```bash
python baseline/baseline.py
```

Funktionen:
- Liest Konfiguration aus `configs/default.yaml`
- Trainiert das Flexible_CNN-Modell
- Speichert das beste Modell nach `model/best_model.pth`

---

### 4. Modelltest

Ausführen:

```bash
python baseline/baseline_test.py
```

Ausgabe:
- Test Loss
- Test Accuracy

---

### 5. Domain Adaptation Training (In Arbeit)

Method DANN:

```bash
method_DANN/m_DANN.py
method_DANN/m_DANN.ipynb
```

Ausgabe:
- Source Accuracy:0.3996
- Test Accuracy:0.6826
- Performanceverbesserung:0.283

Method pseudo:

```bash
method_pseudo/m_pseudo.py
method_pseudo/iterative_pseudo_labeling.py
```

Ausgabe:

- Source Accuracy:
- Test Accuracy:
- Performanceverbesserung:





---

## Konfigurationsdatei (configs/default.yaml)

```yaml
baseline:
  batch_size: 16
  learning_rate: 0.0001694841438362755
  weight_decay: 0.0006531754995659995
  num_layers: 6
  kernel_size: 15
  start_channels: 8
  num_epochs: 30
  early_stopping_patience: 5
```

---

## Modellarchitektur

- Modell befindet sich in `models`
- Parameter anpassbar: Anzahl der Convolution-Layers, Startkanäle, Kernelgröße, Aktivierungsfunktionen

---

## TODO

- [ ] Vergleich mehrerer Modelle
- [ ] TensorBoard-Integration
- [ ] Confusion Matrix-Ausgabe
- [ ] Flexible Daten-Typumwandlung


