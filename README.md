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

#### Link utili

[Link Tutorial Utile](https://github.com/sara-kassani/Medical-Image-Processing/blob/master/2.%20Kaggle-Full%20Preprocessing%20Tutorial.ipynb)
[Link Tool Dataset](https://github.com/dvschultz/dataset-tools)

## Deep Learning

Fine tuning su [Swim Transformer](https://github.com/microsoft/Swin-Transformer)

### Perché Swim Transformer?

| Caratteristica                      | Swim Transformer                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| Tipo di Patch                       | Gerarchico, con finestra scorrevole                                                      |
| Gestione della scala                | Ottima                                                                                   |
| Prestazioni su immagini mediche     | Spesso migliori, più dettagli visivi rispetto a ViT-base-patch16                       |
| Efficienza computazionale           | Maggiore di ViT-base-patch16                                                             |
| Generalizzazione su piccoli dataset | Spesso superiore (utilizzato perlopiù con dataset piccoli) rispetto a ViT-base-patch16 |

## Studenti

[Antonio Colamartino](https://github.com/Tony0380) | a.colamartino6@studenti.uniba.it

[Claudio De Candia](https://github.com/ClaudideCandia) | c.decandia15@studenti.uniba.it
