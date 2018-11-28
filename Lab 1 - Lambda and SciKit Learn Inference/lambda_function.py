import json
import boto3
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

warnings.filterwarnings(action='ignore', category=Warning) 
s3 = boto3.resource('s3')

# make Prediction on the trained model
def lambda_handler (event, context):

	# Bucket_name is passed on from event
	data = json.loads(json.dumps(event))
	
	# Initialize constants 
	BUCKET_NAME = data['bucket_name']
	DATADIR='ServerlessAIWorkshop/SentimentAnalysis'
	MODEL_FILE = 'sentiment_analysis_model.pkl'
	VOCAB_FILE = 'vocabulary.pkl'
	MODEL_FILE_KEY = DATADIR + '/' + MODEL_FILE
	VOCAB_FILE_KEY = DATADIR + '/' + VOCAB_FILE
	LOCAL_MODEL_FILE = '/tmp/' +  MODEL_FILE
	LOCAL_VOCAB_FILE = '/tmp/' +  VOCAB_FILE
	TEST_TWEET = data['test_tweet']
	
	# get trained model/pkl file from S3 bucket
	s3.Object(BUCKET_NAME, MODEL_FILE_KEY).download_file(LOCAL_MODEL_FILE)
	s3.Object(BUCKET_NAME, VOCAB_FILE_KEY).download_file(LOCAL_VOCAB_FILE)

	# loading the trained model as well as test data and label
	model = joblib.load(LOCAL_MODEL_FILE) 
	vocabulary_to_load =joblib.load(LOCAL_VOCAB_FILE)
	loaded_vectorizer = CountVectorizer(vocabulary=vocabulary_to_load)

	transformed_tweets = loaded_vectorizer.transform([TEST_TWEET])
	transformed_tweets = transformed_tweets.toarray()

	sentiment = model.predict(transformed_tweets)
	results = {'sentiment': sentiment[0] }
	
	return {
        "statusCode": 200,
        "body": json.dumps(results)
    }
