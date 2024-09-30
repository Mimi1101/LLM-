import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import os


def extract_5000_questions():
    """
    A function to extract first 5000 questions from the dev-v2.0.json
    """
    with open('dev-v2.0.json', 'r')  as f:
        data = json.load(f)
    
    questions = []

    #Iterating through the dataset to extract qestions

    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])
                # Stop if we've collected 5000 questions
                if len(questions) == 5000:
                    break
            if len(questions) == 5000:
                break
        if len(questions) == 5000:
            break
    print(f"Extracted {len(questions)} questions:")
    for question in questions[:10]:  # Display first 10 questions for preview
        print(question)

# Save the extracted questions to a file if needed
    with open('first_5000_questions.json', 'w') as output_file:
        json.dump(questions, output_file)


def GPTResponses():
    """
    A function for the gpt-4o-mini to answer the first 5000 questions using batch api
    """
    load_dotenv()

    with open('first_5000_questions.json', 'r') as f:
        questions = json.load(f)
    # for question in enumerate(questions, 1):
    #     print(f"question {question}")

    system_prompt = "You are an INTELLIGENT ASSISTANT who excels at answering questions on wide range of topics . Your job is to ANSWER THE QUESTION CORRECTLY, PROPERLY, and, TO THE BEST OF YOUR ABILITY"
    user_prompt = "The question you will answer is {question}"
    
    tasks = []
    for idx, question in enumerate(questions, 1):
        messsages = [
            {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(question=question)}
        ]

        custom_id = f"question_{idx}"

        task = {
            "custom_id": custom_id,
             "method": "POST",
             "url": "/v1/chat/completions",
             "body": {
                "model" : "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messsages 
             }
        }
        tasks.append(task)

    # Correctly writing tasks to a JSONL file
    input_file_path = 'batch_input_tasks.jsonl'
    with open(input_file_path, 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    
    #Uploading the batch input file
    batch_file = client.files.create(
        file=open(input_file_path, 'rb'),
        purpose='batch'
    )
    print(f"Uploaded batch file: {batch_file.id}")

    #Creating a batch job
    batch_job = client.batches.create( 
        input_file_id = batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Created batch job: {batch_job.id}")

    #Checking the batch job status
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        else:
            time.sleep(5)

    print("Batch processing completed.")

    #Getting the output file once the batch is collected
    output_file_id = check.output_file_id
    result = client.files.content(output_file_id).content

    # Save the results to a file
    output_file_path = 'batch_output_openai2.jsonl'
    with open(output_file_path, 'wb') as file:
        file.write(result)

   

if __name__ == "__main__":
   GPTResponses()
