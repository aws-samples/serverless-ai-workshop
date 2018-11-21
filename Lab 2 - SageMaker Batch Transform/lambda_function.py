import warnings
import boto3
import json
import sagemaker

#Warning is the base class so it should catch all types of warnings
warnings.filterwarnings(action='ignore', category=Warning) 
s3 = boto3.resource('s3')

# make Prediction on the trained model
def lambda_handler(event, context):

	# Bucket_name and model name are passed on from event
	data = json.loads(json.dumps(event))
	BUCKET_NAME = data['bucket_name']
	MODEL_NAME = data['model_name']
	
	# Initialize onstants variables
	VALIDATION_FILE = 'validation_data.csv'
	VALIDATION_KEY = 'ServerlessAIWorkshop/data/' + VALIDATION_FILE
	PREDICTION_FOLDER = 'ServerlessAIWorkshop/prediction/'
	PREDICTION_KEY = PREDICTION_FOLDER + VALIDATION_FILE + ".out"
	INPUT_URL = 's3://{}/{}'.format(BUCKET_NAME, VALIDATION_KEY)
	OUTPUT_URL = 's3://{}/{}'.format(BUCKET_NAME, PREDICTION_FOLDER)
	
	# Initialize the transformer object
	transformer =sagemaker.transformer.Transformer(
		base_transform_job_name='Batch-Transform',
		model_name=MODEL_NAME,
		instance_count=1,
		instance_type='ml.c4.xlarge',
		output_path=OUTPUT_URL
		)
	
	# To start a transform job:
	transformer.transform(INPUT_URL, content_type='text/csv', split_type='Line')

	# Then wait until transform job is completed
	transformer.wait()
	
	
	return {
        "statusCode": 200,
        "body": json.dumps('Hello from Lambda!')
    }