# %%
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

df = pd.read_csv("YoutubeCommentsDataSet.csv")
df.loc[len(df)] = ["not bad at all", "positive"] 
comments = df["Comment"].to_numpy().astype(str)
vectorizer = keras.layers.TextVectorization(max_tokens = 15000, output_sequence_length=50, output_mode="int")
vectorizer.adapt(comments)
vectorized_comments = vectorizer(comments)
vectorized_comments = vectorized_comments.numpy()
labels = df["Sentiment"]
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(vectorized_comments, encoded_labels, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(200,)),
    keras.layers.Embedding(15000, 32),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=False)),  # Bi-LSTM layer
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(3, activation = "softmax"),
])


class_labels = np.unique(encoded_labels)
class_weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=encoded_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

optimizer = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=20, class_weight=class_weight_dict)



# %%
x = vectorizer("not the best")
x = np.expand_dims(x.numpy(), axis=0)

prediction = model.predict(x)
predicted_label = np.argmax(prediction) 
decoded_sentiment = encoder.inverse_transform([predicted_label])  # Convert index to original labe
print(prediction)
print(decoded_sentiment)
# %%
