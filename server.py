from flask import Flask, request, jsonify
from flask_cors import CORS
from SentimentalAnalysis import main
from CAH import predict as predictCAH
from NeuralNetworks import predict as predictNN
import logging

app = Flask(__name__)
CORS(app)

# App routes
@app.route('/', methods=['POST'])
def home():
    return "Welcome to LogInsight's DataMining model API"

@app.route('/SentimentalAnalysis/', methods=['POST'])
def SentimentalAnalysis():
    return main()

@app.route('/PredictCluster/', methods=['POST'])
def PredictCluster():
    try:
        data = request.get_json(force=True)
        dataVolume = data.get('dataVolume')
        jobDuration = data.get('jobDuration')
        nbrComponent = data.get('nbrComponent')

        if not all([dataVolume, jobDuration, nbrComponent]):
            raise ValueError('Missing required fields in JSON payload')

        # Call your predict function here
        predicted_cluster = predictCAH(jobDuration, dataVolume,nbrComponent )
        predicted_cluster = int(predicted_cluster)  # convert int64 to integer

        return jsonify({'dataVolume': dataVolume, 'jobDuration': jobDuration, 'nbrComponent': nbrComponent, 'predicted_cluster': predicted_cluster})

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)})
    
@app.route('/PredictDuration/', methods=['POST'])
def PredictDuration():
    try:
        data = request.get_json(force=True)
        dataVolume = data.get('dataVolume')
        nbrComponent = data.get('nbrComponent')
        
        if not all([dataVolume, nbrComponent]):
            raise ValueError('Missing required fields in JSON payload')

        # Call your predict function here
        jobDuration = predictNN(dataVolume,nbrComponent)
        jobDuration = int(jobDuration)  # convert int64 to integer

        return jsonify({'dataVolume': dataVolume, 'jobDuration': jobDuration, 'nbrComponent': nbrComponent})

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)