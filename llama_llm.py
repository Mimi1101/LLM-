import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from dotenv import load_dotenv

def llamaResponses():
    """
    A function for the llama 3.1 8b model to answer questions serially.
    """
    # Load environment variables
    load_dotenv()

    # Initialize the ChatCompletionsClient
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

    # Load the questions from the JSON file
    with open('first_5000_questions.json', 'r') as f:
        questions = json.load(f)

    # defining the system prompt
    system_prompt = "You are an INTELLIGENT ASSISTANT who excels at answering questions on wide range of topics . Your job is to ANSWER THE QUESTION CORRECTLY, PROPERLY, and, TO THE BEST OF YOUR ABILITY"
    
    # Open the output file
    output_file_path = 'llama_azure.jsonl'
    with open(output_file_path, 'w') as outfile:
        # Process each question serially
        for idx, question in enumerate(questions, 1):
            user_prompt = f"The question you will answer is: {question}"
            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ]

            try:
                # Get the response from the model
                response = client.complete(messages=messages)

                # Extract the answer
                answer = response.choices[0].message.content

                # Prepare the result dictionary
                result = {
                    "question_idx": idx,
                    "question": question,
                    "answer": answer,
                }

                # Write the result to the output file
                outfile.write(json.dumps(result) + '\n')

                print(f"Processed question {idx}")

            except Exception as e:
                print(f"Error processing question {idx}: {e}")


if __name__ == "__main__":
    llamaResponses()
