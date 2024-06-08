import json
import boto3
import nltk
import langchain
import re
from sentence_transformers import SentenceTransformer
from langchain.llms import BedrockModel
from langchain.pipelines import SentenceSegmentation, TransformersNER
from nltk.translate.bleu_score import sentence_bleu

# Define AWS credentials (replace with yours)
aws_access_key_id = " "
aws_secret_access_key = " "
region_name = " "

# Define model ID
model_id = "amazon.titan-text-express-v1"

# Load pre-trained models
sentence_model = SentenceTransformer("all-mpnet-base-v2")
ner_model = TransformersNER("distilbert-base-cased-finetuned-conll03-english")
summarization_model = BedrockModel(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name,
    model_id=model_id
)


def get_meeting_text_from_s3(bucket_name, object_key):
    """
    Fetches text data from a specified S3 object.

    Args:
        bucket_name (str): Name of the S3 bucket containing the meeting transcript.
        object_key (str): Key (filename) of the S3 object with the transcript text.

    Returns:
        str: The text content of the S3 object, or None if download fails.
    """
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def clean_text(text):
    """
    Preprocesses text for summarization.
    
    Args:
        text (str): The meeting transcript text.
    
      Returns:
        str: The cleaned text.
    """
    
    # Lowercase text
    cleaned_text = text.lower()
    
    # Remove punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
  
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def load_reference_summaries(filepath):
  """
  Loads reference summaries from a text file.

  Args:
      filepath (str): Path to the text file containing reference summaries.

  Returns:
      list: List of reference summaries (strings).
  """
  try:
    with open(filepath, 'r') as f:
      summaries = f.readlines()
    return [summary.strip() for summary in summaries]
  except FileNotFoundError:
    print(f"Error: Reference summary file not found at {filepath}")
    return []

def summarize_meeting(meeting_text):
    # Preprocess text 
    cleaned_text = clean_text(meeting_text)

    # Construct input data for Bedrock model
    input_data = {
        "inputText": cleaned_text,
        "textGenerationConfig": {
            "temperature": 0.7,  # Adjust as needed (0.7 for factual summaries)
            "topP": 0.9,       # Adjust as needed (0.9 for high-confidence summaries)
            "maxTokenCount": 256,  # Adjust as needed (256 for concise summaries)
        }
    }

    # Send request to Bedrock runtime and parse response
    response = summarization_model.generate_text(body=json.dumps(input_data))
    results = response["results"]
  
    # Extract summary text
    summary_text = results[0]["outputText"].strip()  # Remove leading/trailing whitespace
  
    # Calculate BLEU score (assuming a single reference summary)
    reference_summaries = load_reference_summaries("filepath")
    bleu_score = sentence_bleu(reference_summaries[0].split(), summary_text.split())
  
    print(f"Summary: {summary_text}")
    print(f"BLEU Score: {bleu_score:.2f}")  # Format score with two decimal places
  
    return summary_text  # Optional: return the summary for further use


def main():
    # Replace with your bucket and object key
    bucket_name = " "
    object_key = "meeting_transcript.txt"

    # Get meeting text from S3
    meeting_text = get_meeting_text_from_s3(bucket_name, object_key)

    # Summarize the meeting
    summary = summarize_meeting(meeting_text)

    print(summary)


if __name__ == "__main__":
    main()