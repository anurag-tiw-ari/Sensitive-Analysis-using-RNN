
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

print("Movie Sentiment Analysis Starting...")


def setup_model():
    """Check and setup model - FIXED VERSION"""
    if not os.path.exists('simple_rnn_imdb.h5'):
        print(" Model not found. Training now...")
        return train_model_fixed()
    try:
        model = load_model('simple_rnn_imdb.h5')
        print(" Model loaded successfully")
        return model
    except:
        print("Corrupted model file. Retraining...")
        return train_model_fixed()

def train_model_fixed():
    """Train the sentiment analysis model - FIXED VERSION"""
    print(" Loading IMDB dataset...")

    max_features = 10000
    max_len = 500

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

   
    print(f" Data Balance - Positive: {np.sum(y_train)}, Negative: {len(y_train) - np.sum(y_train)}")

    print("🔧 Preprocessing data...")
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    print(" Building model architecture...")
   
    model = Sequential()
    model.add(Embedding(max_features, 64, input_length=max_len))  # Reduced from 128
    model.add(SimpleRNN(64, activation='tanh', dropout=0.2, recurrent_dropout=0.2))  # Added dropout
    model.add(Dense(1, activation='sigmoid'))

  
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(" Training model (this takes 2-5 minutes)...")
  
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=10,  # Increased epochs
        batch_size=128,  # Larger batch size
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    if final_train_acc < 0.7:  
        print(" Training didn't go well. Using alternative approach...")
        return train_model_alternative()

    print("Saving model...")
    model.save('simple_rnn_imdb.h5')
    print("Model trained successfully!")
    return model

def train_model_alternative():
    """Alternative training approach if first fails"""
    print("🔄 Trying alternative training approach...")

    max_features = 10000
    max_len = 500

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(max_features, 32, input_length=max_len))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )

    model.save('simple_rnn_imdb.h5')
    return model

word_index = imdb.get_word_index()

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)

def predict_sentiment(review, model):
    """Predict sentiment for a given review"""
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    score = prediction[0][0]
    sentiment = 'Positive' if score > 0.5 else 'Negative'
    confidence = score if score > 0.5 else 1 - score
    return sentiment, confidence, score

def quick_model_test(model):
    """Test if model is working properly"""
    print("\n Running Quick Model Test...")

    test_cases = [
        ("excellent amazing wonderful fantastic superb brilliant", "CLEAR_POSITIVE"),
        ("terrible awful horrible disgusting waste boring", "CLEAR_NEGATIVE"),
        ("the movie", "NEUTRAL")
    ]

    all_correct = True

    for text, expected in test_cases:
        sentiment, confidence, score = predict_sentiment(text, model)

        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Got: {sentiment} (score: {score:.4f})")

        # Check if prediction makes sense
        if expected == "CLEAR_POSITIVE" and sentiment != "Positive":
            print("WRONG PREDICTION!")
            all_correct = False
        elif expected == "CLEAR_NEGATIVE" and sentiment != "Negative":
            print("WRONG PREDICTION!")
            all_correct = False

        print("-" * 40)

    if all_correct:
        print("Model is working correctly!")
    else:
        print("Model has issues - predictions don't make sense")

    return all_correct


def main():
    # Load model
    print("🚀 Loading model...")
    model = setup_model()

    if not quick_model_test(model):
        print("\n Model failed basic test. Retraining...")
        model = train_model_alternative()
        if not quick_model_test(model):
            print("❌ Model still broken. There might be a fundamental issue.")
            return

    print("\n" + "="*50)
    print(" MOVIE SENTIMENT ANALYZER READY!")
    print("="*50)

    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the story was engaging from start to finish.",
        "Terrible movie. Complete waste of time. Poor acting, boring plot, and awful direction.",
        "It was okay, nothing special but not bad either. Some good moments but overall forgettable.",
        "I loved this film! The characters were well-developed and the cinematography was stunning.",
        "The worst movie I've ever seen. The plot made no sense and the acting was wooden."
    ]

    print("\n Testing with example reviews:")
    print("-" * 50)

    for i, review in enumerate(test_reviews, 1):
        sentiment, confidence, score = predict_sentiment(review, model)
        print(f"\nExample {i}:")
        print(f"Review: {review[:60]}...")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw Score: {score:.4f}")

        # Quick validation
        positive_words = ['fantastic', 'superb', 'loved', 'stunning']
        negative_words = ['terrible', 'waste', 'boring', 'awful', 'worst']

        has_positive = any(word in review.lower() for word in positive_words)
        has_negative = any(word in review.lower() for word in negative_words)

        if has_positive and sentiment == "Negative":
            print("  Suspicious: Positive words but negative prediction")
        elif has_negative and sentiment == "Positive":
            print("  Suspicious: Negative words but positive prediction")

        print("-" * 50)

    print("\n INTERACTIVE TESTING")
    print("Type 'quit' to exit")
    print("-" * 40)

    while True:
        user_review = input("\nEnter a movie review: ")
        if user_review.lower() == 'quit':
            break

        if user_review.strip():
            sentiment, confidence, score = predict_sentiment(user_review, model)
            print(f"\n RESULTS:")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Score: {score:.4f}")

            bar_length = int(confidence * 20)
            print(f"Confidence: [{'█' * bar_length}{'░' * (20 - bar_length)}] {confidence:.1%}")
        else:
            print("Please enter a review.")

if __name__ == '__main__':
    main()