# SmaLLM – Ein Sprachmodell von Grund auf
*Wichtig! Dieses README ist zu Teil von KI-Generiert aber ich Arbeite noch an der Überarbeitung*
## Wie ich gearbeitet habe

Dieses Projekt entstand als Bottom-Up Lernprojekt: Jedes Modul wurde einzeln verstanden, geschrieben und getestet – bevor das nächste dazukam. Keine Copy-Paste Lösungen, kein Framework das die Arbeit übernimmt. Der Fokus lag darauf zu verstehen *warum* jede Zeile so ist wie sie ist, nicht nur *dass* sie funktioniert.

Reihenfolge: `config.py` → `model.py` (Head für Head) → `data.py` → `train.py` → `generate.py`. Jede Klasse wurde im `playground.ipynb` mit Dummy-Inputs getestet bevor sie in den nächsten Schritt eingebaut wurde.

---

## Projektstruktur

```
mini-LLM/
├── src/
│   ├── config.py        # Alle Hyperparameter
│   ├── model.py         # Transformer-Architektur
│   ├── data.py          # Tokenizer + Dataset
│   ├── train.py         # Trainingsloop
│   ├── generate.py      # Textgenerierung
│   └── playground.ipynb # Tests
├── checkpoint/          # Letzte 4 Checkpoints
├── epochs/              # Checkpoint pro Epoch
└── data/                # Lokale Trainingsdaten (optional)
```

---

## Architektur

### model.py

#### AttentionHead

Ein einzelner Attention-Kopf berechnet für jeden Token wie stark er auf andere Tokens achten soll. Das ist der Kern des Kontextverständnisses .

```
Input:  (batch, seq_len, EMBEDDING_DIM)
Output: (batch, seq_len, HEAD_DIM)
```

Ablauf:
1. Query, Key, Value aus dem Input projizieren (je eine Linear-Schicht)
2. Attention Scores: `Q @ K^T / sqrt(HEAD_DIM)`
3. Causal Mask: zukünftige Tokens werden auf `-inf` gesetzt (das Modell darf nicht "schummeln")
4. Softmax → Attention Weights
5. Output: `Weights @ V`

#### MultiHeadAttention

Mehrere Heads laufen parallel – jeder lernt andere Muster (Syntax, Semantik, Referenzen). Die Outputs werden zusammengeführt und durch eine Projektion gemischt.

```
Input:  (batch, seq_len, EMBEDDING_DIM)
Output: (batch, seq_len, EMBEDDING_DIM)
```

#### TransformerBlock

Pre-Norm Architektur: LayerNorm kommt *vor* Attention und MLP, nicht danach. Residual Connections sorgen dafür dass der Gradient gut fliesst.

```python
x = x + dropout(attention(layernorm(x)))  # Attention-Pfad
x = x + dropout(mlp(layernorm(x)))        # MLP-Pfad
```

Das MLP (4× EMBEDDING_DIM) verarbeitet Information *pro Token* – während Attention Tokens miteinander vergleicht, "denkt" das MLP über jeden Token einzeln nach.

#### SmaLLM

Das Gesamtmodell:

```
Token-Indizes (batch, seq_len)
    → Token Embedding + Position Embedding
    → N × TransformerBlock
    → LayerNorm
    → Linear → Logits (batch, seq_len, VOCAB_SIZE)
```

Position Embeddings sind gelernt (nicht sinusoidal) – das Modell lernt selbst wie es Positionen codiert.

---

### data.py

#### Tokenizer

Statt character-level (65 Zeichen) wird `tiktoken` mit `cl100k_base` verwendet – derselbe Tokenizer wie GPT-4. Ein Token entspricht ca. 3–4 Zeichen, häufige Wörter werden ein einzelner Token.

```python
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Hallo Welt")  # → [39, 6316, ...]
```

#### TextDataset

Das Dataset lädt Artikel von HuggingFace (Wikipedia DE) und tokenisiert sie zu einem langen Tensor. `__getitem__` schneidet Chunks der Länge `CONTEXT_LEN` heraus:

```
Input:  data[i : i + CONTEXT_LEN]
Target: data[i+1 : i + CONTEXT_LEN + 1]
```

Das Target ist immer um einen Token verschoben – das Modell lernt den *nächsten* Token vorherzusagen.

---

### train.py

#### Trainingsloop

- **Optimizer:** AdamW mit lr=3e-4
- **Loss:** CrossEntropyLoss über alle Token-Positionen
- **Mixed Precision:** `autocast` + `GradScaler` für schnelleres Training auf GPU (bfloat16)
- **Gradient Clipping:** `clip_grad_norm_(..., 1.0)` verhindert explodierende Gradienten
- **Gradient Accumulation:** 4 Steps akkumuliert = effektiv größerer Batch


### generate.py

Autoregressive Generierung: Das Modell gibt Logits für alle Positionen aus, aber nur der letzte Token (`logits[0, -1, :]`) wird für die Vorhersage verwendet. Softmax → multinomial Sampling → neuer Token wird angehängt. Wiederholen bis `max_new_tokens` erreicht.

---