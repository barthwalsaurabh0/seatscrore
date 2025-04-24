import pickle
import pandas as pd

# Function to load the model and make predictions
def predict_seatscore(age, gender, fatigue):
    # Load the saved model
    with open("seatscore_tree.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Fatigue': [fatigue]})
    
    # Predict
    prediction = model.predict(input_data)
    
    # Cap the prediction between 0 and 100
    return min(max(prediction[0], 0), 100)

# Example usage:
# seat_score = predict_seatscore(30, 1, 50)
# print(f"Predicted SeatScore: {seat_score}")
