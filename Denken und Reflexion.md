### Denken und Reflexion：

1. Für die **Baseline** habe ich verschiedene **Datenvorverarbeitungsmethoden** getestet: MinMax-Skalierung, Z-Score-Normalisierung und einen Normalizer (angepasst an den Quelldaten, gemeinsam für Ziel verwendet). Leider führten diese nicht zu guten Ergebnissen auf dem Testset. Der Grund könnte die Abhängigkeit von globalen Statistiken sein, daher verwende ich nun die normalisierenden Schichten des Netzwerks selbst.

2. Beim **Feature-Extractor** hat sich nach mehreren Versuchen bestätigt, dass **GroupNorm** die besten Ergebnisse liefert. Die Normalisierung erfolgt über halb so viele Gruppen wie Schichten, im Vergleich zu BatchNorm (batch-abhängig) und LayerNorm (schichtenweise). GroupNorm vermeidet Batch-Statistiken - Mittelwert und Varianz werden kanalgruppenweise berechnet, was die Unterscheidbarkeit zwischen Kanälen erhält.

3. Für den **Klassifikationskopf** wurde kein einfacher Classifier verwendet, sondern eine Kombination aus LayerNorm-Normalisierung und Cosine-Ähnlichkeitsclassifier. Ein Temperature-Parameter skaliert die Logits und beeinflusst die Softmax-Konfidenzverteilung, was sich für Domain Adaptation eignet.

4. Bei **Pseudolabels** führte das direkte Generieren und Trainieren wegen der hohen Rauschrate (60% falsche Labels bei 40% Vorhersagegenauigkeit) nur zu ~5% Verbesserung. In späteren Iterationen overfittet das Modell oft auf das Rauschen. Geplant ist die Kombination mit DANN unter Verwendung höherqualitatiger Pseudolabels.

5. Bei **DANN** lagen **die Anpassungen hauptsächlich am Klassifikationskopf und Feature-Extractor**, da das ursprüngliche Modell zu einfach war: Unzureichende Feature-Diskriminierung und fehlende Domain-Alignment führten zu keiner Verbesserung. Nach den Änderungen zeigte sich deutlicher Fortschritt. Ein weiteres DANN-Problem ist das **Stoppkriterium** - ohne Ziel-Labels muss es anhand interner Metriken entscheiden. Die Monitoring der Vorhersagegenauigkeit auf Ziel-Daten zeigt oft zuerst Verbesserung, dann Verschlechterung. **Aktuell wird anhand von Trainingsiterationen, Quell-Genauigkeit und Domain-Alignment-Grad entschieden** (gemessen an der Abweichung der Domain-Diskriminator-Genauigkeit von 50% Zufallstrefferquote).

6. **Diese Methode hat Schwächen**: Sie beruht nur auf Quell-Metriken und dem Domain-Diskriminator, nicht auf echter Ziel-Datenperformance. Feste Schwellenwerte (z.B. gap < 0.001) sind unflexibel. **Geplant**: Nutzung von Pseudolabel-Konfidenzverteilungen und Feature-Trennanalyse (z.B. Cluster-Silhouetten-Scores，Feature MMD).

```
gap = abs(dom_acc - 0.5)
if gap < 0.01 and avg_cls_loss < 0.05 and epoch > 10:
    patience +=1
    if gap < best_gap:
        best_gap = gap
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"[INFO] patience {patience} / 3")
        if patience > 3:
            model.load_state_dict(best_model_state)
            print("[INFO] Early stopping: domain aligned and classifier converged.")
            break
    else:
        patience = 0
        best_gap = gap
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
```















