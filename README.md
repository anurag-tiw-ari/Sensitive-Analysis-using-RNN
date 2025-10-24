#  Movie Review Sentiment Analysis

A small, easy-to-run deep learning project that classifies movie reviews as **positive** or **negative** using a Recurrent Neural Network (RNN). Built with TensorFlow, Keras and deployed with Streamlit for a simple interactive web UI.

---

##  Highlights

* **Real-time sentiment analysis** through a Streamlit front-end.
* **RNN-based model** with embedding layer for sequence representation.
* **Auto-training behaviour**: if a trained model (`simple_rnn_imdb.h5`) is not present, the app trains one automatically.
* **Designed for reproducibility** — works locally or on Google Colab.

---

##  Features

* Instant sentiment classification (Positive / Negative) with a confidence score
* Preprocessing pipeline: tokenization, sequence padding/truncation
* Dropout and early stopping to reduce overfitting
* Easy to extend — swap `SimpleRNN` for `LSTM`/`GRU` or upgrade to transformer models (BERT) later

---

##  Technologies

* **TensorFlow / Keras** — model building & training
* **Streamlit** — interactive web app
* **NumPy** — numerical operations
* **IMDB Dataset** (Keras-built-in) — 50,000 labelled reviews
* **Google Colab** — optional development + free GPU

---

##  Project Structure

```
movie-sentiment-analysis/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (this file)
├── simple_rnn_imdb.h5     # Trained model (auto-generated)
└── assets/                # Screenshots and demo images
```

---

##  Installation

1. Clone the repo

```bash
git clone https://github.com/anurag-tiw-ari/Sensitive-Analysis-using-RNN.git
cd Sensitive-Analysis-using-RNN
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

> If `simple_rnn_imdb.h5` is missing the app will train a model on the IMDB dataset automatically — training time depends on your machine.

---

##  Usage

1. Start the app with `streamlit run app.py`.
2. Enter a movie review in the text box.
3. Click **Analyze Sentiment**.
4. The app returns `Positive` or `Negative` with a confidence score.

### Example reviews to try

* "This movie was absolutely fantastic! Great acting and storyline."
* "Terrible waste of time. Poor acting and boring plot."
* "It was okay, nothing special but not bad either."

---

##  Model Details (example configuration)

* **Vocabulary size**: 10,000 tokens
* **Embedding dimension**: 128
* **Sequence length**: 200 tokens (padding/truncation)
* **RNN layer**: `SimpleRNN` with 128 units (activation: `tanh`)
  *(You can replace with `LSTM` or `GRU` for better performance on longer dependencies.)*
* **Output**: Single neuron with `sigmoid` activation for binary classification
* **Training**: default example uses `epochs=3` and a `validation_split=0.2` (adjust as needed)

---

## Performance & Behavior

* Clear positive/negative reviews: **high confidence (≈85–95%)**
* Mixed/neutral reviews: **lower confidence (≈50–75%)**
* Use a larger model (LSTM/BERT) or more data / fine-tuning for better edge-case performance


---

##  Troubleshooting & Tips

* **Model trains very slowly**: use Colab or a GPU. Reduce `epochs` or `batch_size` to speed up experimentation.
* **App crashes with memory errors**: reduce `maxlen` (sequence length) or `vocab_size`, or run training on Colab.
* **Low accuracy on your custom reviews**: try increasing dataset size, switching to `LSTM`, or fine-tuning a pretrained transformer.

---

##  Quick Colab Start

1. Open Google Colab.
2. Upload the repository or mount your GitHub.
3. Install `requirements.txt` and run the notebook cells or `app.py`.

---

##  Future Enhancements

* Multi-language support (tokenizers / language models)
* Replace SimpleRNN with `LSTM` / `GRU` or a Transformer model (BERT)
* Add batch-processing / CSV import for bulk predictions
* Add explainability (LIME / SHAP) to show which words drove the prediction
* Provide model versioning + small web API for programmatic access

---



