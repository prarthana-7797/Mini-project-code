import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset (example data)
def load_data(file_path):
    return pd.read_csv(file_path)


# Preprocessing
def preprocess_data(data):
    x = data.drop(columns=['Light_Status'])  # Features
    y = data['Light_Status']  # Target variable
    return x, y


# Train Decision Tree classifier
def train_model(x_train, y_train):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf


# Evaluate accuracy
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


# Function to make decision based on motion sensor input
def make_decision(model, motion_sensor_data):
    # Assuming motion_sensor_data is a dictionary with keys as feature names
    prediction = model.predict([list(motion_sensor_data.values())])[0]
    return prediction


# Main function
def main():
    # Load dataset
    data = load_data('motion_sensor_data.csv')

    # Preprocess data
    x, y = preprocess_data(data)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(x_train, y_train)

    # Evaluate model
    evaluate_model(model, x_test, y_test)

    # Example usage
    new_data = {'Motion_Sensor_1': 1, 'Motion_Sensor_2': 0, 'Time_of_day': 'evening'}  # Example motion sensor data
    decision = make_decision(model, new_data)
    print("Decision:", decision)


if __name__ == "__main__":
    main()
