from openai import AzureOpenAI
from openai import OpenAI
import re
import subprocess
import tempfile
import os
import requests
from tqdm import tqdm
import time



def process_texts_with_api(sentences, model_type):
    """
    Extract GO terms from sentences using Azure OpenAI API.
    
    Args:
        sentences (list of str): List of sentences to process
        model_type (str): Model type to use (e.g., 'gpt-4o')
    
    Environment Variables:
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
        AZURE_OPENAI_KEY: Azure OpenAI API key
        AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name (default: 'gpt-4o')
        AZURE_OPENAI_API_VERSION: API version (default: '2024-12-01-preview')
    
    Returns:
        list of list: List of GO term lists for each sentence
    """
    # Read from environment variables
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', 
                              'https://egmm-meuu4s6i-swedencentral.cognitiveservices.azure.com/')
    subscription_key = os.environ.get('AZURE_OPENAI_KEY', '')
    deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'o4-mini')
    api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    
    if not subscription_key:
        raise ValueError("AZURE_OPENAI_KEY environment variable is not set")
    
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    terms_list = []
    with tqdm(total=len(sentences), desc="Processing", unit="sentence") as pbar:# only related to molecular function. "The annotation may include direct descriptions (e.g., FUNCTION section) or related descriptions (e.g., SIMILARITY, DOMAIN section)."
        for sentence in sentences:#"Identify related GO terms, even if they are not explicitly stated in the description."
            try:
            # Create the prompt input with system and user messages   "Identify GO terms that are related to this description.\n" "You should infer GO terms not only from direct mentions, but also from the related biological domain knowledge. "
                # prompt_input = [
                #     {
                #         "role": "system",
                #         "content": (
                #             "You are a bioinformatics expert specializing in Gene Ontology (GO) term extraction. "
                #             "Your task is to comprehensively extract Gene Ontology (GO) terms based on the provided description. "
                #             "Use biological knowledge to identify GO terms even if they are not explicitly stated."
                #         )
                #     },
                #     {
                #         "role": "user",
                #         "content": (
                #             f"description: \"{sentence}\"\n\n"
                #             "List all GO terms that are relevant to this description.\n"
                #             "Respond in the following format: 'GO:XXXXXXX; GO:XXXXXXX'. "
                #             "If no GO terms are applicable, return an empty string ''."
                #         )
                #     }
                # ]


                # completion = client.chat.completions.create(
                #     model=model_type,  # e.g. gpt-35-instant
                #     messages=prompt_input,
                # )
                # completion = client.chat.completions.create(
                #     messages=prompt_input,
                #     max_completion_tokens=10000,
                #     model=deployment
                # )
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a bioinformatics expert specializing in Gene Ontology (GO) term extraction. Your task is to comprehensively extract Gene Ontology (GO) terms based on the provided description. Use biological knowledge to identify GO terms even if they are not explicitly stated.",
                        },
                        {
                            "role": "user",
                            "content": f"description: \"{sentence}\""
                            "List all GO terms that are relevant to this description."
                            "Respond in the following format: 'GO:XXXXXXX; GO:XXXXXXX'. "
                            "If no GO terms are applicable, return an empty string ''.",
                        }
                    ],
                    max_tokens=1024,
                    temperature=1.0,
                    top_p=1.0,
                    model=deployment
                )
                # print(type(sentence)


                extracted_terms = response.choices[0].message.content#.content
                # Use regex to extract valid GO terms (GO: followed by 7 digits)
                terms = re.findall(r'GO:\d{7}', extracted_terms)
                terms = list(set(terms))

            except Exception as e:
                # Log error and clear terms on failure
                print(f"[Warning]: {e}")
                terms = []
                
            terms_list.append(terms)
            print('go_terms',terms)
            pbar.update(1)

    return terms_list




def process_texts_with_ollama(sentences, model_type):

    terms_list = []
    with tqdm(total=len(sentences), desc="Processing", unit="sentence") as pbar:
        for sentence in tqdm(sentences):
            prompt = (
                "You are a bioinformatics expert specializing in Gene Ontology (GO) term extraction. "
                "Your task is to comprehensively extract Gene Ontology (GO) terms based on the provided description. "
                "Use biological knowledge to identify GO terms even if they are not explicitly stated."
                f"Sentence: \"{sentence}\"\n\n"
                "List all GO terms that are relevant to this description.\n"
                "Respond in the following format: 'GO:XXXXXXX; GO:XXXXXXX'. "
                "If no GO terms are applicable, return an empty string ''."
            )

            # Send to Ollama local server
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_type,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                content = response.json()["response"]
                terms = re.findall(r'GO:\d{7}', content)
                terms = list(set(terms))
            else:
                print(f"Error: {response.status_code}")
                terms = []

            terms_list.append(terms)
            print('go_terms',terms)
            pbar.update(1)

    return terms_list
























# def check_ollama_running(host="127.0.0.1", port=11434):
#     """
#     Check if the Ollama service is running.

#     Args:
#         host (str): Host where Ollama service is running.
#         port (int): Port where Ollama service is running.

#     Returns:
#         bool: True if the service is running, False otherwise.
#     """
#     url = f"http://{host}:{port}"
#     try:
#         response = requests.get(url, timeout=5)
#         return response.status_code == 200
#     except requests.RequestException:
#         return False
def get_ollama_models():
    """Retrieve the list of models available in Ollama and count them."""
    try:
        result = subprocess.run(
            ["/project/cyang7/ollama_models/bin/ollama", "list"], 
            capture_output=True, text=True, check=True
        )
        models = result.stdout.strip().split("\n")
        
        # Exclude the header if present
        if models and "MODEL" in models[0]:
            models = models[1:]
        
        return models, len(models)
    except subprocess.CalledProcessError as e:
        print("Error running 'ollama list':", e)
        return [], 0



def run_ontogpt_extract(input_text, model):
    """ollama/llama3.2
    Run the 'ontogpt extract' command on the input text.

    Args:
        input_text (str): Text to process.
        model (str): Model name for ontogpt extract.

    Returns:
        str: Raw output from the ontogpt command.
    """
    # Ensure Ollama service is running, start it if not
    # if not check_ollama_running():
    #     print("Ollama service is not running. Attempting to start it...")





    # command = '/project/cyang7/ollama_models/bin/ollama serve > ollama_serve_11434.log'

    # command = 'curl http://127.0.0.1:11434'

    # subprocess.Popen(command, shell=True)


    # command = 'lsof -iTCP -sTCP:LISTEN -P | grep ollama'
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    print(f'The model {model} is running.')






    # try:
    #     response = requests.get("http://localhost:11660", timeout=5)
    #     if response.status_code == 200:
    #         print("Ollama service is running.")
    #     else:
    #         raise RuntimeError(f"Unexpected response status code: {response.status_code}")
    # except requests.RequestException as e:
    #     raise RuntimeError(f"Failed to start Ollama service: {e}")






        # with open("ollama_serve.log", "a") as log:
            

        # Wait for the service to start
    #     for _ in range(10):  # Wait for up to 10 seconds
    #         if check_ollama_running():
    #             print("Ollama service is now running.")
    #             break
    #         time.sleep(1)
    #     else:
    #         raise RuntimeError("Failed to start Ollama service within the timeout period.")

    # # Confirm Ollama service is active via curl
    # if not check_ollama_running():
    #     raise RuntimeError("Ollama service is not running as expected after startup attempt.")

    # Create a temporary file to store input text
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmpfile:
        tmpfile.write(input_text)
        tmpfile.flush()
        tmp_filename = tmpfile.name

    api_command = ["runoak", "set-apikey", "-e", "openai", "930b565487df47fd8898b7cfb5490ca7"]    
    api_result = subprocess.run(api_command, capture_output=True, text=True)   

    try:
        # Run the ontogpt extract command
        command = ["ontogpt", "extract", "-i", tmp_filename, "-t", "go_terms", '--api-base', 'https://egm-openai-2.openai.azure.com/',  "-m", model]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        return result.stdout
    # except Exception as e:
    #     print("Unexpected error:", str(e))
    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def extract_go_terms_from_output(ontogpt_output):
    """
    Extract GO terms from the ontogpt output.

    Args:
        ontogpt_output (str): Raw output from ontogpt.

    Returns:
        list: A list of extracted GO terms.
    """
    # Adjusted regex to match 'molecularfunctions' section and GO terms
    go_regex = re.compile(r"molecularfunctions:\s*(-.*?)(?:\n\w|$)", re.DOTALL)
    term_regex = re.compile(r"GO:\d+")

    match = go_regex.search(ontogpt_output)
    if match:
        # Extract the part under 'molecularfunctions:'
        molecularfunctions_text = match.group(1)
        # Find all GO terms in the extracted text
        return term_regex.findall(molecularfunctions_text)
    return []





def extract_all_unique_go_terms(ontogpt_output):
    """
    Extract all unique GO terms from the ontogpt output.

    Args:
        ontogpt_output (str): Raw output from ontogpt.

    Returns:
        list: A sorted list of unique GO terms.
    """
    # Regex to find all GO terms (e.g., GO:1234567)
    term_regex = re.compile(r"GO:\d+")
    
    # Find all GO terms in the output
    go_terms = term_regex.findall(ontogpt_output)
    
    # Remove duplicates by converting to a set, then sort the terms
    unique_go_terms = sorted(set(go_terms))
    
    return unique_go_terms





def process_texts_with_ontogpt(texts, model_type):
    """
    Process a list of texts with ontogpt and extract GO terms.

    Args:
        texts (list of str): List of input texts to process.
        model (str): Model name for ontogpt extract.

    Returns:
        list of list: A list of lists, where each sublist contains GO terms for each input text.
    """
    results = []
    with tqdm(total=len(texts), desc="Processing texts", unit="text") as pbar:
        for text in texts:
            ontogpt_output = run_ontogpt_extract(text, model=model_type)
            print(ontogpt_output)
            go_terms = list(set(extract_go_terms_from_output(ontogpt_output)))
            # print('___________')
            # # # print('text',text)
            # print('ontogpt_output',ontogpt_output)
            # # print('@@@@@@')
            print('go_terms',go_terms)
            # print('___________')
            results.append(go_terms)
            pbar.update(1)

    # print(results)
    return results














# def process_single_text(text, model="ollama/llama3.2"):
#     """
#     Process a single text using ontogpt and extract GO terms.

#     Args:
#         text (str): Text to process.
#         model (str): Model name for ontogpt extract.

#     Returns:
#         list: Extracted GO terms from the text.
#     """
#     ontogpt_output = run_ontogpt_extract(text, model=model)
#     go_terms = extract_go_terms_from_output(ontogpt_output)
#     return go_terms




# def process_texts_with_multithreading(texts, model="ollama/llama3.2", max_workers=4):
#     """
#     Process a list of texts using multithreading.

#     Args:
#         texts (list of str): List of input texts to process.
#         model (str): Model name for ontogpt extract.
#         max_workers (int): Number of threads to use.

#     Returns:
#         list of list: A list of lists, where each sublist contains GO terms for each input text.
#     """
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_single_text, text, model) for text in texts]
#         for future in tqdm(futures, desc="Processing texts", unit="text"):
#             results.append(future.result())
#     return results








def main():
    """
    Main function to process input texts and extract GO terms using ontogpt.
    """
    # Example input lists
    all_predictions = [
        'Combining with the neurotransmitter dopamine and activating adenylate cyclase via coupling to Gi/Go to initiate a change in cell activity.',
        "Catalyzes the hydrolysis of 6-phosphogluconolactone to 6-phosphogluconate. Involved in the regulation of cellular redox state; enzymatic activity is required for this function. Required for sugar-dependent expression of nitrate assimilation genes in the nucleus of root cells.",
        "Catalyzes the reversible isomerization of citrate to isocitrate via cis-aconitate. Involved in the catabolism of short chain fatty acids (SCFA) via the tricarboxylic acid (TCA)(acetyl degradation route) and probably via the 2-methylcitrate cycle I (propionate degradation route).",
        "Its catalytic efficiency is greatest for aldehydes, suggesting the reduction of aromatic and medium-chain aliphatic aldehydes is its in vivo activity. Prefers NADPH to NADH. Active on aliphatic compounds up to 5 carbons in length and aromatic alcohols, less effective on branched-chain primary alcohols. Active on a wide variety of primary alcohols and their corresponding aldehydes, but not against ketones nor secondary alcohols. Plays a role in tolerance to internally produced ethanol."
    ]

    all_references = [
        "Involved in the regulation of nutrient metabolism. Is associated with a DNA binding complex that binds to the G box, a well-characterized cis-acting DNA regulatory element found in plant genes.",
        "Involved in the regulation of nutrient metabolism. Is associated with a DNA binding complex that binds to the G box, a well-characterized cis-acting DNA regulatory element found in plant genes. Negative regulator of freezing tolerance that modulates cold-responsive C-repeat-binding factors (CBF) DREB1A AND DREB1B proteins stability by facilitating their ubiquitin-mediated degradation; this processus is counteracted by B1L."
    ]

    # Process both lists
    print("Processing predictions...")
    predicted_go_terms = process_texts_with_ontogpt(all_predictions)
    

    print("Processing references...")
    reference_go_terms = process_texts_with_ontogpt(all_references)

    # Print results
    print("Predicted GO Terms:", predicted_go_terms)
    print("Reference GO Terms:", reference_go_terms)

if __name__ == "__main__":
    main()