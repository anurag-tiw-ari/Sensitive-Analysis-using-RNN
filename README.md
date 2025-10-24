``` markdown
# Movie Review Sentiment Analysis

A deep learning project that classifies movie reviews as positive or negative using Recurrent Neural Networks (RNN). Built with TensorFlow, Keras, and Streamlit.



## Features

- **Real-time Sentiment Analysis**: Instant classification of movie reviews  
- **Deep Learning Model**: RNN-based architecture with word embeddings  
- **Interactive Web Interface**: User-friendly Streamlit application  
- **High Accuracy**: Strong performance on positive/negative classification  
- **Auto-Training**: Model trains automatically if not present  

## Technologies Used

- **TensorFlow & Keras** - Deep learning framework  
- **Streamlit** - Web application deployment  
- **NumPy** - Numerical computations  
- **IMDB Dataset** - 50,000 movie reviews for training  
- **Google Colab** - Development environment  

## Installation

1. **Clone the repository**
   ```bash
   https://github.com/anurag-tiw-ari/Sensitive-Analysis-using-RNN.git
   
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

##  Usage

1. **Launch the app** using `streamlit run app.py`
2. **Enter a movie review** in the text area
3. **Click "Analyze Sentiment"** to get instant classification
4. **View results** including sentiment (Positive/Negative) and confidence score

### Example Reviews to Try:

* "This movie was absolutely fantastic! Great acting and storyline."
* "Terrible waste of time. Poor acting and boring plot."
* "It was okay, nothing special but not bad either."

##  Project Architecture

```
Input Text → Preprocessing → Word Embeddings → RNN Layer → Output Classification
```

### Model Details:

* **Embedding Layer**: 10,000 vocabulary → 128 dimensions
* **RNN Layer**: 128 units with ReLU activation
* **Output Layer**: Sigmoid activation for binary classification
* **Training**: 3 epochs on IMDB dataset with 20% validation split

## Performance

* **Clear Positive Reviews**: 85-95% confidence
* **Clear Negative Reviews**: 85-95% confidence
* **Mixed Reviews**: Appropriate uncertainty (50-75% confidence)
* **Overall Performance**: Strong classification on diverse review types

## Model Optimization

The project demonstrates several machine learning best practices:

* **Hyperparameter Tuning**: Optimized learning rate, batch size, and layer dimensions
* **Regularization**: Dropout layers to prevent overfitting
* **Early Stopping**: Prevents overtraining and finds optimal training point
* **Proper Validation**: Train/validation splits for reliable performance measurement

##  Example Output

```
Review: "This movie was fantastic! Great acting and amazing story."
Result: POSITIVE (92% confidence)

Review: "Terrible movie. Waste of time and money."
Result: NEGATIVE (96% confidence)

Review: "It was okay, nothing special."
Result:  NEGATIVE (74% confidence)
```

##  Project Structure

```
movie-sentiment-analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── simple_rnn_imdb.h5     # Trained model (auto-generated)
└── assets/                # Screenshots and demo images
```

##  Quick Start with Google Colab

This project was developed in Google Colab for easy experimentation:

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the project files
3. Run the code with free GPU acceleration
4. Experience faster training times (2-5 minutes vs 15+ minutes locally)

## Key Learnings

* **End-to-end ML Pipeline**: From data preprocessing to model deployment
* **NLP Challenges**: Handling variable text lengths, word embeddings, sequence processing
* **Model Optimization**: Hyperparameter tuning and regularization techniques



## Future Enhancements

* Add support for multiple languages
* Implement more advanced models (LSTM, BERT)
* Add batch processing for multiple reviews
* Include confidence threshold adjustments
* Add model explainability features

