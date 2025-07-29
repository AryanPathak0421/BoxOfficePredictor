import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from predictor import TicketSalesPredictor  # ✅ Import from separate file

# Load dataset
df = pd.read_csv("movies_metadata.csv", low_memory=False)

# Clean and select relevant columns
df = df[['budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'revenue']].copy()
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df.dropna(inplace=True)
df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Target: estimated ticket sales
average_ticket_price = 10
df['ticket_sales'] = df['revenue'] / average_ticket_price

# Features and target
X = df[['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']]
y = df['ticket_sales']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Use imported wrapper class
predictor = TicketSalesPredictor(model)

# Save model using joblib
joblib.dump(predictor, "ticket_sales_predictor.joblib")
print("✅ Model saved as ticket_sales_predictor.joblib")




# train.py

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import joblib
# from predictor import TicketSalesPredictor

# # ... rest of the training code ...


# # Load dataset
# df = pd.read_csv("movies_metadata.csv", low_memory=False)

# # Clean and select relevant columns
# df = df[['budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'revenue']].copy()
# df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
# df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
# df.dropna(inplace=True)
# df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# # Target: estimated ticket sales
# average_ticket_price = 10
# df['ticket_sales'] = df['revenue'] / average_ticket_price

# # Features and target
# X = df[['budget', 'runtime', 'popularity', 'vote_average', 'vote_count']]
# y = df['ticket_sales']

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Create a prediction wrapper
# class TicketSalesPredictor:
#     from predictor import TicketSalesPredictor  # ✅ keep this

#     def __init__(self, model):
#         self.model = model

#     def predict_ticket_sales(self, movie_data):
#         features = [
#             float(movie_data.get('budget', 0)),
#             float(movie_data.get('runtime', 0)),
#             float(movie_data.get('popularity', 0)),
#             float(movie_data.get('vote_average', 0)),
#             float(movie_data.get('vote_count', 0))
#         ]
#         return self.model.predict([features])[0]

# # Save the model
# predictor = TicketSalesPredictor(model)
# joblib.dump(predictor, "ticket_sales_predictor.joblib")
# print("✅ Model saved as ticket_sales_predictor.joblib")
