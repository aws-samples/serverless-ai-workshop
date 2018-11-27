from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import boto3
import random
import json

#Warning is the base class so it should catch all types of warnings
warnings.filterwarnings(action='ignore', category=Warning) 
s3 = boto3.resource('s3')

# make Prediction on the trained model
def lambda_handler(event, context):

	# Bucket_name is passed on from event
	data = json.loads(json.dumps(event))
	BUCKET_NAME = data['bucket_name']
	TEST_TWEET = data['test_tweet']
	
	# Initialize onstants variables
	PICKLE_FILE_NAME = '/tmp/sentiment_analysis_tweet.pkl'
	VOCAB_FILE_NAME = '/tmp/vocabulary.pkl'
	PICKLE_FILE_KEY = 'ServerlessAIWorkshop/SentimentAnalysis/sentiment_analysis_tweet.pkl'
	VOCAB_FILE_KEY = 'ServerlessAIWorkshop/SentimentAnalysis/vocabulary.pkl'

	# get trained model/pkl file as well as vocabulary from S3 bucket
	s3.Object(BUCKET_NAME, PICKLE_FILE_KEY).download_file(PICKLE_FILE_NAME)
	s3.Object(BUCKET_NAME, VOCAB_FILE_KEY).download_file(VOCAB_FILE_NAME)
	
	# loading the trained model as well as test data and label
	model = joblib.load(PICKLE_FILE_NAME) 
	vocabulary_to_load =joblib.load(VOCAB_FILE_NAME)
	loaded_vectorizer = CountVectorizer(vocabulary=vocabulary_to_load)

	transformed_tweets = loaded_vectorizer.transform([TEST_TWEET])
	transformed_tweets = transformed_tweets.toarray()
	
	# predict using the test dataset
	y_pred = model.predict(transformed_tweets)
	
	return {
		"statusCode": 200,
		"body": json.dumps('The resulted sentiment is ' + y_pred[0])
	}
