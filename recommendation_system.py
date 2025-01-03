import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystem:
    def __init__(self, data):
        """
        Initialize the Recommendation System with the provided dataset.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing at least 'id' and 'description' columns.
        """
        self.data = data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(data['description'])

    def recommend(self, item_id, top_n=5):
        """
        Recommend top N similar items to the given item ID.

        Parameters:
        - item_id (int): The ID of the item for which recommendations are needed.
        - top_n (int): The number of recommendations to return.

        Returns:
        - List[Dict]: A list of recommended items with their IDs and similarity scores.
        """
        if item_id not in self.data['id'].values:
            raise ValueError(f"Item ID {item_id} not found in the dataset.")

        item_idx = self.data[self.data['id'] == item_id].index[0]
        similarity_scores = cosine_similarity(self.tfidf_matrix[item_idx], self.tfidf_matrix).flatten()
        
        top_indices = similarity_scores.argsort()[-top_n-1:-1][::-1]
        recommendations = [
            {
                'id': self.data.iloc[idx]['id'],
                'description': self.data.iloc[idx]['description'],
                'score': similarity_scores[idx]
            }
            for idx in top_indices
        ]

        return recommendations

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'description': [
            "AI-based stock prediction",
            "Financial planning application",
            "AI-driven portfolio management",
            "Stock market analysis tools",
            "Personalized financial advisory system"
        ]
    })

    recommender = RecommendationSystem(data)
    recommendations = recommender.recommend(item_id=1, top_n=3)
    print("Recommendations:")
    for rec in recommendations:
        print(f"ID: {rec['id']}, Description: {rec['description']}, Score: {rec['score']:.2f}")

