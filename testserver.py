from flask import Flask, request, jsonify
from SentimentalAnalysis import main

app = Flask(__name__)

# load example model
#model = joblib.load('example_model.pkl')

# define endpoint for model prediction
@app.route('/SentimentalAnalysis/', methods=['POST'])
def SentimentalAnalysis():
    return main()

if __name__ == '__main__':
    app.run(debug=True)

"""'Rate', 'Date', 'Product_Name', 'User_Function', 'Company_Size', 'Industry', 'Title', 'Text', 'Sentiment', 'Sentiment_Category']"""
