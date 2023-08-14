from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

def load_data():
    data = pd.read_csv("iris.csv")
    return data

data = load_data()
data = data.drop('Id', axis=1)

# Split the data into features (X) and target (y)
X = data.drop("Species", axis=1)
y = data["Species"]

# Etiketleri sayısal değerlere dönüştürün
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Dense Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),  # 4 özellik var
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 çiçek türü var
])

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Gelen verileri bir veri çerçevesine dönüştürme
        # Make predictre
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(features)

        # Tahmin sonuçlarını 1D diziye dönüştürme
        predicted_classes = np.argmax(prediction, axis=1)

        predicted_species = label_encoder.inverse_transform(predicted_classes)[0]


        return render_template('index.html', prediction=predicted_species)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
