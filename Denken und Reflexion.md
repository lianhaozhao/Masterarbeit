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













1. 对于baseline 我分别进行了 minmax,zscore以及normalizer(source 上拟合后传进来，源/目标共用)的数据处理，但遗憾的是并没有在测试集上取得好的结果，可能的原因可能是依赖全局统计量，所以采用网络本身的归一化

2. 而对特征提取部分，在经过多次尝试，确认Groupnorm 会取得较好的结果，根据layer的层数的一半去归一化数据，相比同批次batchnorm 和 同 layer层 layernorm，不依赖 Batch 统计量：计算均值和方差时，按通道分组计算，保留通道间的区分性。

3. 对于分类头，没有采用原来简单的分类器而是结合了 LayerNorm 归一化 和 Cosine 相似度分类器，添加temperature控制 logits 的缩放程度，影响 softmax 的置信度分布 适用于领域自适应

4. 对于pseudo,在直接生成pseudo，在进行训练，因为60%的伪标签是错误的，即噪声占比远高于有效信号（预测准确率0.4）。可能会出现经过几轮迭代后，模型可能完全拟合噪声。所以提升有限大约在5%，后续和结合DANN等方法 一起使用基于更高的准确率的pseudo

5. 对于DANN 调整基本专注在分类头和特征提取器，因为一开始的模型过于简单，特征判别性不足，领域对齐失败，基本没有什么提升，在修改分类头和特征提取器后，效果明显好转。DANN 的另一个问题：何时判断停止，因为目标数据没有标签，只能使用本身的数据进行判断。（ 经过监控在训练期间的目标数据集的预测准确率，发现会出现：先提高在降低的情况。）我采用根据一下方法：训练次数，源数据的准确率和领域对齐程度 （ 根据接近领域判别器准确率与随机猜测水平的绝对偏差，越接近零，表示域无法区分。）

   这种方法存在一定的问题：早停决策完全依赖源域指标（分类损失）和域判别器，并不是真实目标域的预测效果,固定阈值（如 gap < 0.001）缺乏适应性

   计划：伪标签置信度分布；特征分布的可分性（如聚类轮廓系数）

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
