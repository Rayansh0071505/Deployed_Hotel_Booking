import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained machine learning model
model = pickle.load(open("model.pkl", "rb"))

# Define a function to make predictions using the loaded model
def predict_cancellation(features):
    prediction = model.predict(features)
    return prediction

def main():
    # Streamlit app title and description
    st.title("Hotel Booking Cancellation Prediction")
    st.write("This app predicts whether a hotel booking is likely to be canceled.")

    # Collect user input for prediction
    hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    lead_time = st.slider("Lead Time (days)", 0, 365, 30)
    arrival_date_year = st.selectbox("Arrival Date Year", [2015, 2016, 2017, 2018])
    arrival_date_month = st.selectbox("Arrival Date Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    arrival_date_week_number = st.slider("Arrival Date Week Number", 1, 53, 27)
    arrival_date_day_of_month = st.slider("Arrival Date Day of Month", 1, 31, 15)
    stays_in_weekend_nights = st.slider("Stays in Weekend Nights", 0, 10, 2)
    stays_in_week_nights = st.slider("Stays in Week Nights", 0, 20, 5)
    adults = st.slider("Number of Adults", 1, 10, 1)
    children = st.slider("Number of Children", 0, 10, 0)
    babies = st.slider("Number of Babies", 0, 5, 0)
    meal = st.selectbox("Meal", ["BB", "FB", "HB", "SC"])
    # st.write("Note: Meal options have been label-encoded for prediction purposes.")
    st.write("BB: Bed & Breakfast (0)")
    st.write("FB: Full Board (1)")
    st.write("HB: Half Board (2)")
    st.write("SC: Self Catering (3)")
    country = st.text_input("Country")
    st.write("Enter your country code")
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary", "Groups"])
    distribution_channel = st.selectbox("Distribution Channel", ["TA/TO", "Direct", "Corporate", "GDS"])
    is_repeated_guest = st.selectbox("Is Repeated Guest", [0, 1])
    previous_cancellations = st.slider("Previous Cancellations", 0, 20, 0)
    previous_bookings_not_canceled = st.slider("Previous Bookings Not Canceled", 0, 20, 0)
    reserved_room_type = st.selectbox("Reserved Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "P", "L"])
    assigned_room_type = st.selectbox("Assigned Room Type", ["A", "B", "C", "D", "E", "F", "G", "H", "P", "L"])
    booking_changes = st.slider("Booking Changes", 0, 10, 0)
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    agent = st.slider("Agent", 0, 536, 0)
    company = st.slider("Company", 0, 543, 0)
    days_in_waiting_list = st.slider("Days in Waiting List", 0, 391, 0)
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
    adr = st.slider("ADR", 0.0, 5400.0, 100.0)
    required_car_parking_spaces = st.slider("Required Car Parking Spaces", 0, 8, 0)
    total_of_special_requests = st.slider("Total Special Requests", 0, 5, 0)
    reservation_status = st.selectbox("Reservation Status", ["Canceled", "Check-Out", "No-Show"])
    reservation_status_date = st.text_input("Reservation Status Date")

    # When the user clicks the "Predict" button
    if st.button("Predict"):
    # Prepare the input features as a numpy array
        input_features = np.array([
            [hotel, lead_time, arrival_date_year, arrival_date_month, arrival_date_week_number,
            arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights,
            adults, children, babies, meal, country, market_segment, distribution_channel,
            is_repeated_guest, previous_cancellations, previous_bookings_not_canceled,
            reserved_room_type, assigned_room_type, booking_changes, deposit_type, agent,
            company, days_in_waiting_list, customer_type, adr, required_car_parking_spaces,
            total_of_special_requests, reservation_status, reservation_status_date]
        ],dtype=object)

        # Label encoding
        label_encoded_features = []
        for i in range(len(input_features[0])):
            if i in [0, 3, 11, 12, 13, 14, 18, 19, 20, 21, 22, 25, 26, 27, 29, 30]:
                # Apply label encoding for relevant features
                labelencoder = LabelEncoder()
                # Ensure input_features[i] is iterable
                if isinstance(input_features[0][i], (list, tuple)):
                    label_encoded_features.append([labelencoder.fit_transform(input_features[0][i])])
                else:
                    label_encoded_features.append(labelencoder.fit_transform([input_features[0][i]]))
            else:
                # For non-categorical features, keep the value as is
                label_encoded_features.append([input_features[0][i]])

        # Reshape back to original format
        flattened_features = [item for sublist in label_encoded_features for item in sublist]
        input_features = np.array(flattened_features).reshape(1, -1)

        # Get the prediction
        prediction = predict_cancellation(input_features)

        if prediction == 1:
            st.write("Prediction: Booking is likely to be canceled.")
        else:
            st.write("Prediction: Booking is not likely to be canceled.")

if __name__ == "__main__":
    main()
