import joblib

model = joblib.load("iris_model.pkl")

def predict_species(features):
    prediction = model.predict([features])[0]
    classes = ["Setosa", "Versicolor", "Virginica"]
    return classes[prediction]

print("Enter flower features:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

features = [sepal_length, sepal_width, petal_length, petal_width]
species = predict_species(features)
print("Predicted species:", species)