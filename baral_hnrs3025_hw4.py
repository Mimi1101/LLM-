import json
import os
import re
import time
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

def create_chroma_db():
    """Creates a Chroma database and adds SQuAD 2.0 contexts."""
    load_dotenv('.env')

    # Load the SQuAD 2.0 dev dataset with UTF-8 encoding
    with open('dev-v2.0.json', 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    # Extract contexts from the dataset
    contexts = []
    for item in squad_data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            contexts.append(context)

    # Generate document IDs for each context   
    doc_ids = [f'doc_{i}' for i in range(len(contexts))]

    # Initialize ChromaDB client and embedding function
    chroma_client = chromadb.PersistentClient(path="my_chromadb")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # Check if the "documents" collection exists, if not, create it
    existing_collections = chroma_client.list_collections()
    if any(c.name == "documents" for c in existing_collections):
        collection = chroma_client.get_collection(name="documents")
    else:
        collection = chroma_client.create_collection(
        name="documents",
        embedding_function=openai_ef
    )
    
    # Add documents (contexts) to the collection
    collection.add(
        documents= contexts,
        ids = doc_ids
    )

def extract_500_questions_and_answers():
    """Extracts the first 500 questions and their corresponding answers from the SQuAD 2.0 dataset."""

    with open('dev-v2.0.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    questions_answers = []
    maximum_questions = 500

    # Extract questions and answers, skipping those marked as "is_impossible"
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            for qa in paragraph["qas"]:
                #Skip the question marked as is_impossible
                if not qa["is_impossible"]:
                    question = qa["question"]

                    #Combining all the valid answers into a list
                    answers = [answer["text"] for answer in qa["answers"]]

                     # Store the question and its answers
                    questions_answers.append({
                    "question": question,
                    "answers": answers
                    })
                     # Stop after extracting 500 questions
                    if len(questions_answers) >= maximum_questions:
                        break
            if len(questions_answers) >= maximum_questions:
                     break
        if len(questions_answers) >= maximum_questions:
            break

    with open("500_questions_answers.json", "w") as output_file:
        json.dump(questions_answers, output_file, indent=4)
    print("Extracted 500 questions and answers.")

def gpt_rag():
    """Generates RAG (Retrieval-Augmented Generation) responses using GPT-4o-mini."""
    load_dotenv()

    #load the questions
    with open('500_questions_answers.json', 'r') as f:
        data = json.load(f)
    # Initialize ChromaDB client and embedding function  
    chroma_client = chromadb.PersistentClient(path="my_chromadb")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_collection(name="documents", embedding_function=openai_ef)

    system_prompt = """
    You are an INTELLIGENT ASSISTANT who EXCELS in answering questions when provided the given contexts for it. USE THE CONTEXT TO ANSWER THE QUESTIONS and ANSWER IT TO THE BEST OF YOUR ABILITY AND CORRECTLY AVOID USING the word CONTEXT in your answer.
    """

    
    tasks = []
    # Create tasks for each question
    for idx, item in enumerate(data, 1):
        question = item["question"]
        query_text = f"Question: {question}"

        # Retrieve top 5 contexts using vector search
        results = collection.query(query_texts=[query_text], n_results=5)
        retrieved_contexts = results['documents'][0]
        context = " ".join(retrieved_contexts)
        user_prompt = f"""Answer the given {question} provided the {context}.
        Ensure your response is CLEAR and DIRECTLY RELATED to the context provided."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Batch task creation
        custom_id = f"question_{idx}"
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messages
            }
        }
        tasks.append(task)
    # Write tasks to a JSONL file
    input_file_path = 'batch_input_tasks_with_RAG.jsonl'
    with open(input_file_path, 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    # Initialize OpenAI client      
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
     # Upload batch input file
    batch_file = client.files.create(
        file=open(input_file_path, 'rb'),
        purpose='batch'
    ) 
    print(f"Uploaded batch file: {batch_file.id}")

    # Create a batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Created batch job: {batch_job.id}")

    # Check batch job status
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        else:
            time.sleep(10)
    print("Batch processing with RAG completed.")


def llama_rag():
    """Generates RAG (Retrieval-Augmented Generation) responses using GPT-4o-mini."""

    load_dotenv()
    # Initialize the ChatCompletionsClient
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )
    #load the questions
    with open('500_questions_answers.json', 'r') as f:
        data = json.load(f)

     # Initialize ChromaDB client and embedding function
    chroma_client = chromadb.PersistentClient(path="my_chromadb")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    collection = chroma_client.get_collection(name="documents", embedding_function=openai_ef)

    system_prompt = """
    You are an INTELLIGENT ASSISTANT who EXCELS in answering questions when provided the given contexts for it. USE THE CONTEXT TO ANSWER THE QUESTIONS and ANSWER IT TO THE BEST OF YOUR ABILITY AND CORRECTLY. AVOID USING the word CONTEXT in your answer.
    """
    output_file_path = 'llama_rag.jsonl'
    with open(output_file_path, 'w') as outfile:
        for idx, item in enumerate(data, 1):
            question = item["question"]
            query_text = f"Question: {question}"
            results = collection.query(query_texts=[query_text], n_results=5)
            retrieved_contexts = results['documents'][0]
            context = " ".join(retrieved_contexts)
            user_prompt = f"""Answer the given {question} provided the {context}.
            Ensure your response is CLEAR and DIRECTLY RELATED to the context provided."""
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
    print("Finished answering questions.")

def extract_answers_from_document_for_scoring_gpt():
    """
    Extracting answers from the dataset in a specific format that can match the etxracted answers from gpt-4o-mini (RAG)
    """
    with open('500_questions_answers.json', 'r') as f:
        data = json.load(f)
    correct_answers = {}
    question_counter = 1
    
    for item in data:
        question_key = f"question_{question_counter}"
        question_counter += 1

         # Extract unique correct answers
        answers = list(set(item['answers']))
        correct_answers[question_key] = answers

        # Save the extracted correct answers into a JSON file
    with open('correct_answers_extracted.json', 'w') as output_file:
        json.dump(correct_answers, output_file, indent=4)

def extract_answers_from_document_for_scoring_llama():
    """
    Extracting answers from the dataset in a specific format that can match the etxracted answers from the llama model (RAG)
    """
    with open('500_questions_answers.json', 'r') as f:
        data = json.load(f)
    correct_answers = {}
    question_counter = 1
    
    for item in data:
        question_key = question_counter
        question_counter += 1

         # Extract unique correct answers
        answers = list(set(item['answers']))
        correct_answers[question_key] = answers

        # Save the extracted correct answers into a JSON file
    with open('_correct_answers_extracted.json', 'w') as output_file:
        json.dump(correct_answers, output_file, indent=4)

  
def extract_answers_from_rag_gpt(path):
    """
    Extracting answers from the gpt-4o-mini batch rag file
    """
    gpt_rag_answers = {}
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    for item in data: 
        question_key = item['custom_id']
        model_answer = item['response']['body']['choices'][0]['message']['content']
        gpt_rag_answers[question_key] = model_answer
    with open('gpt_rag_answers_extracted.json', 'w') as output_file:
            json.dump(gpt_rag_answers, output_file, indent=4)
    
    
def extract_answers_from_llama_rag(path):
    """
    Extracting answers from llama rag file
    """
    llama_rag_answers = {}
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    for item in data:
        question_key = item['question_idx']
        model_answer = item['answer']
        llama_rag_answers[question_key] = model_answer
    with open('llama_rag_answers_extracted.json', 'w') as output_file:
            json.dump(llama_rag_answers, output_file, indent=4)


def llama_scoring():
    """
    checks if the llama rag got the correct answer by comparing it with the corrct answer from the dataset.
    """
    load_dotenv()

    # Correct answers
    with open('llama_correct_answers_extracted.json', 'r') as f:
        correct_answers = json.load(f)
        
    # LLama model answers
    with open('llama_rag_answers_extracted.json', 'r') as f2:
        gpt_answers = json.load(f2)

    scoring_prompt = """
    You are a teacher tasked with determining whether a student’s answer to a question was correct, 
    based on a set of possible correct answers. You must only use the provided possible correct answers 
    to determine if the student’s response was correct. 
    Question: {question} 
    Student’s Response: {student_response} 
    Possible Correct Answers: 
    {correct_answers} 
    Your response should only be a valid Json as shown below: 
    {{ 
    “explanation” (str): A short explanation of why the student’s answer was correct or 
    incorrect., 
    “score” (bool): true if the student’s answer was correct, false if it was incorrect 
    }} 
    Your response:
    """

    tasks = []

    for question_key, correct_answer_list in correct_answers.items():
        student_response = gpt_answers.get(question_key)

        if not student_response:
            continue 
        
        # Formatting the scoring prompt for the current question
        prompt = scoring_prompt.format(
            question=question_key,
            student_response=student_response,
            correct_answers=correct_answer_list
        )
        
        # Creating a task for GPT-4o to evaluate the answer
        task = {
            "custom_id": question_key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that evaluates student answers."},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        tasks.append(task)

    input_file_path = 'llama_rag_scoring_tasks.jsonl'

    with open(input_file_path, 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch_file = client.files.create(
        file=open(input_file_path, 'rb'),
        purpose='batch'
    )
    print(f"Uploaded batch file: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Created batch job: {batch_job.id}")

    # Check the status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        else:
            time.sleep(15)

    print("Batch processing completed.")


def gpt_scoring():
    """
    checks if the gpt-4o-mini rag got the correct answer by comparing it with the corrct answer from the dataset.
    """
    load_dotenv()

    # Correct answers
    with open('correct_answers_extracted.json', 'r') as f:
        correct_answers = json.load(f)
        
    # GPT-4o mini answers
    with open('gpt_rag_answers_extracted.json', 'r') as f2:
        gpt_answers = json.load(f2)

    scoring_prompt = """
    You are a teacher tasked with determining whether a student’s answer to a question was correct, 
    based on a set of possible correct answers. You must only use the provided possible correct answers 
    to determine if the student’s response was correct. 
    Question: {question} 
    Student’s Response: {student_response} 
    Possible Correct Answers: 
    {correct_answers} 
    Your response should only be a valid Json as shown below: 
    {{ 
    “explanation” (str): A short explanation of why the student’s answer was correct or 
    incorrect., 
    “score” (bool): true if the student’s answer was correct, false if it was incorrect 
    }} 
    Your response:
    """

    tasks = []

    for question_key, correct_answer_list in correct_answers.items():
        student_response = gpt_answers.get(question_key)

        if not student_response:
            continue 
        
        # Formatting the scoring prompt for the current question
        prompt = scoring_prompt.format(
            question=question_key,
            student_response=student_response,
            correct_answers=correct_answer_list
        )
        
        # Creating a task for GPT-4o to evaluate the answer
        task = {
            "custom_id": question_key,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are an AI assistant that evaluates student answers."},
                    {"role": "user", "content": prompt}
                ]
            }
        }
        tasks.append(task)

    input_file_path = 'gpt_rag_scoring_tasks.jsonl'

    with open(input_file_path, 'w') as jfile:
        for task in tasks:
            jfile.write(json.dumps(task) + '\n')

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    batch_file = client.files.create(
        file=open(input_file_path, 'rb'),
        purpose='batch'
    )
    print(f"Uploaded batch file: {batch_file.id}")

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Created batch job: {batch_job.id}")

    # Check the status of the batch job
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        else:
            time.sleep(15)

    print("Batch processing completed.")

def jsonl_to_json_GPT(input_file, output_file):

    """Converts a JSONL file into a JSON file, extracting explanations and scores from gpt batch score evaluation output."""
    
    # Read the JSONL file and parse each line
    with open(input_file, 'r', encoding='utf-8') as jsonl_file:
        data = []
        for line in jsonl_file:
            entry = json.loads(line)
            
            # Extract the necessary information
            custom_id = entry.get("custom_id")
            response_content = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content")
            
            # Parse the response content to extract explanation and score
            if response_content:
                response_data = json.loads(response_content)
                explanation = response_data.get("explanation", "")
                score = response_data.get("score", False)
                
                # Format the output
                data.append({
                    "custom_id": custom_id,
                    "explanation": explanation,
                    "score": score
                })
    
    # Write the formatted data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2)


def remove_code_blocks(text):
    # Use regular expressions to remove code block markers
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text.strip()

def convert_jsonl_to_json(input_path, output_path):
    """Converts a JSONL file into a JSON file, extracting explanations and scores from llama  score evaluation output."""
    
    result = []
    
    # Read the JSONL file
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Parse each line as JSON
            data = json.loads(line)
            
            # Initialize a new dictionary for the current line
            entry = {
                'custom_id': data['custom_id'],  # Extract 'custom_id' from the data
                'explanation': None,
                'score': None,
            }
            
            # Get the content string
            content = data['response']['body']['choices'][0]['message']['content']
            content = content.replace('\\(', 'LATEX_START').replace('\\)', 'LATEX_END')
            
            # Remove code block markers if present
            content = remove_code_blocks(content)
            
            try:
                # Try to parse the content as JSON
                content_json = json.loads(content)
                entry['explanation'] = content_json.get('explanation')
                entry['score'] = content_json.get('score')
            except json.JSONDecodeError:
                print(f"Error parsing JSON for result: {data['custom_id']}")
            
            # Append the processed entry to the result list
            result.append(entry)
    
    # Save the processed results
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=2)



def calculate_average(path):
    """Calculates and prints the average score from the scoring results in a JSON file."""
    correct_count = 0
    total_answers = 0
    with open(path, 'r') as file:
        results = json.load(file)
    
    for result in results:
        if result['score'] is not None:
            total_answers += 1
            if result['score'] == True:
                correct_count += 1

    if total_answers > 0:
        average_score = correct_count / total_answers
        print(f"Total correct answers: {correct_count}")
        print(f"Average score for GPT-4o-mini: {average_score}")

        





    
if __name__ == "__main__":

    create_chroma_db()
    extract_500_questions_and_answers()
    gpt_rag()
    llama_rag
    extract_answers_from_document_for_scoring_gpt()
    extract_answers_from_rag_gpt()
    gpt_scoring()
    jsonl_to_json_GPT('gpt_score.jsonl', 'gpt-4o-mini-RAG-10-8-24-hw4.json' )
    calculate_average(path='gpt-4o-mini-RAG-10-8-24-hw4.json')
    extract_answers_from_document_for_scoring_llama()
    extract_answers_from_llama_rag(path='llama_rag.jsonl')
    llama_scoring()
    convert_jsonl_to_json('llama_score.jsonl', 'llama-8B-instruct-RAG-10-9-24-hw4.json')
    calculate_average(path = 'llama-8B-instruct-RAG-10-9-24-hw4.json' )




   