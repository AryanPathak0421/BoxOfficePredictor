class TicketSalesPredictor:
    def __init__(self, model):
        self.model = model

    def predict_ticket_sales(self, movie_data):
        features = [
            float(movie_data.get('budget', 0)),
            float(movie_data.get('runtime', 0)),
            float(movie_data.get('popularity', 0)),
            float(movie_data.get('vote_average', 0)),
            float(movie_data.get('vote_count', 0))
        ]
        return self.model.predict([features])[0]
