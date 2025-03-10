import os  
import base64
import json
import time

from collections import deque
from pocketsphinx import LiveSpeech
from openai import AzureOpenAI  # make sure your import matches your Azure OpenAI SDK

endpoint = os.getenv("ENDPOINT_URL")  
deployment = os.getenv("DEPLOYMENT_NAME")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI Service client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

# Define the system prompt. (Note: In many implementations the "content" field is a string.)
chat_prompt = [
    {
        "role": "system",
        "content": "You are an AI assistant that helps us keep the meeting on track. Use the following chunk of text to suggest if the meeting is staying on track and if not, alert the user by saying so:"
    }
]

# A deque to hold (timestamp, recognized_phrase) tuples.
# This will serve as our sliding window for the last 3 minutes (180 seconds).
last_three_minutes = deque()

# Constant for our window duration in seconds
WINDOW_DURATION = 180

print("Listening for speech...")
for phrase in LiveSpeech():
    # Get the current time (as seconds since epoch)
    current_time = time.time()
    
    # Convert the recognized phrase to a string 
    current_phrase = str(phrase)
    print("Recognized: ", current_phrase)
    
    # Append the current phrase with a timestamp
    last_three_minutes.append((current_time, current_phrase))
    
    # Now remove any phrases that are older than 3 minutes.
    while last_three_minutes and (current_time - last_three_minutes[0][0]) > WINDOW_DURATION:
        last_three_minutes.popleft()
    
    # Join all phrases in the last 3 minutes to form a transcript.
    transcript = " ".join(text for timestamp, text in last_three_minutes)
    print("Transcript for analysis (last 3 minutes):")
    print(transcript)
    
    # Prepare the messages for the OpenAI API by including the transcript as a user message.
    messages = chat_prompt + [{"role": "user", "content": transcript}]
    
    # Generate the completion.
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    
    # Print the API response (in JSON format).
    print(completion.to_json())
    
    # (Optional) Depending on how often you want to send feedback, you might insert a delay here.
    time.sleep(1)