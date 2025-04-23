# SkinVisionNet

SkinVisionNet: Deep Learning Techniques for Accurate Skin Lesion Classification

## Dataset

### Struttura Dataset

|                                | Training | Validation | Test |
| :----------------------------: | -------- | ---------- | ---- |
|       **Melanoma**       | 374      | 30         | 117  |
|        **Nevus**        | 1372     | 78         | 393  |
| **Seborrheic keratosis** | 254      | 42         | 90   |

[Link Dataset Kaggle](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection/)

> ? [possibile secondo dataset](https://challenge.isic-archive.com/data/)

#### Normalizzazione

Standard ImageNet, perfetta per modelli pre-addestrati

### Sbilanciamento

| Tecnica                | Serve per...                              | Quando usarla                        |
|------------------------|-------------------------------------------|--------------------------------------|
| **Data Augmentation**  | Aumentare la varietà e robustezza         | Sempre, su dataset piccoli/medi     |
| **Weighted Sampler**   | Bilanciare il numero di esempi per classe | Quando il dataset è sbilanciato     |
| **Focal Loss**         | Penalizzare di più gli errori gravi       | Quando il modello ignora classi minori |

#### Link utili

[Link Tutorial Utile](https://github.com/sara-kassani/Medical-Image-Processing/blob/master/2.%20Kaggle-Full%20Preprocessing%20Tutorial.ipynb)
[Link Tool Dataset](https://github.com/dvschultz/dataset-tools)

## Algoritmi di Deep Learning

Fine tuning su [Swin Transformer](https://github.com/microsoft/Swin-Transformer) piuttosto che ViT-base-patch16

### Perché Swin Transformer?

| Caratteristica                         | ViT-base-patch16-224                            | Swin Transformer                                     |
|---------------------------------------|--------------------------------------------------|------------------------------------------------------|
| **Tipo di patch**                     | Patch fisse 16x16                                | Finestra scorrevole (shifted windows)               |
| **Architettura**                      | Transformer standard                             | Gerarchica (multi-scala)                            |
| **Gestione della scala**              | Limitata (non gerarchica)                        | Ottima per immagini con dettagli locali             |
| **Prestazioni su immagini mediche**  | Molto buone                                      | Spesso superiori nei task con dettagli fini         |
| **Efficienza computazionale**         | Buona, ma meno efficiente                        | Più efficiente a parità di prestazioni              |
| **Robustezza su piccoli dataset**     | Discreta, con buon fine-tuning                   | Alta, migliore generalizzazione                     |
| **Supporto e documentazione**         | Ampio (HuggingFace, PyTorch, torchvision, ecc.)  | In crescita, ottimo supporto su mmseg e timm        |
| **Facilità d'uso per inizio progetto**| Semplice e veloce da usare                       | Richiede più configurazione                         |
| **Ideale per...**                     | Progetti con risorse limitate o dataset medi     | Classificazione con immagini complesse o dettagliate|

### In sintesi

| Fattore                             | Considerazione                           |
|------------------------------------|------------------------------------------|
| 🔢 **Numero totale immagini**       | ~2660 immagini: ✅ OK                     |
| ⚖️ **Sbilanciamento**               | Presente (ma gestibile)                  |
| 🔬 **Compito visivo complesso**     | Sì, serve riconoscere dettagli           |
| 🧠 **Necessità di contesto globale**| Sì, pattern e bordi nei/lesioni          |
| ⚙️ **Potenza computazionale richiesta** | Media-alta                             |

## Studenti

[Antonio Colamartino](https://github.com/Tony0380) | a.colamartino6@studenti.uniba.it

[Claudio De Candia](https://github.com/ClaudideCandia) | c.decandia15@studenti.uniba.it
