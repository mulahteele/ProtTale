import os
import re

from anthropic import Anthropic
from openai import AzureOpenAI
from tqdm import tqdm


def process_texts_with_api(sentences):
    """
    Extract GO terms from sentences using the Anthropic API.

    Args:
        sentences (list of str): List of sentences to process.

    Environment Variables:
        ANTHROPIC_API_KEY: Anthropic API key.

    Returns:
        list of list: List of GO term lists for each sentence.
    """
    subscription_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not subscription_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    client = Anthropic(api_key=subscription_key)

    terms_list = []
    with tqdm(total=len(sentences), desc="Processing", unit="sentence") as pbar:
        for sentence in sentences:
            try:
                message = client.messages.create(
                    max_tokens=1024,
                    system=("You are a cautious bioinformatics curator for Gene Ontology (GO). "
                            "Goal: map protein descriptions to GO terms ONLY when the description provides clear support."
                            "Prefer high precision over recall.\n\n"
                            "Rules:\n"
                            "1) Only output GO IDs you are highly confident (>=0.8) are correct matches.\n"
                            "2) If the description is too vague or ambiguous, return an empty string: ''.\n"
                            "3) Output MUST be a semicolon-separated list of GO IDs only: "
                            "'GO:XXXXXXX; GO:XXXXXXX' or ''. No other text.\n"
                            "4) Before finalizing, verify that each GO term is directly supported by words/phrases "
                            "in the description; remove any that are not."),
                    messages=[
                        {"role": "user", "content": (f"description:\n\"{sentence}\"\n\n"
                                                     "return only the GO IDs that are directly supported by this description")}
                    ],
                    model="claude-opus-4-6",
                )

                extracted_terms = message.content[0].text
                terms = re.findall(r'GO:\d{7}', extracted_terms)
                terms = list(set(terms))

            except Exception as e:
                print(f"[Warning]: {e}")
                terms = []

            terms_list.append(terms)
            print('go_terms', terms)
            pbar.update(1)

    return terms_list


def process_texts_for_ec_with_api(sentences, model_type):
    """
    Extract EC numbers from sentences using the Azure OpenAI API.

    Args:
        sentences (list of str): List of sentences to process.
        model_type (str): Unused; kept for backwards compatibility.

    Environment Variables:
        AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.
        AZURE_OPENAI_KEY: Azure OpenAI API key.
        AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name (default: 'o4-mini').
        AZURE_OPENAI_API_VERSION: API version (default: '2024-12-01-preview').

    Returns:
        list of list: List of EC number lists for each sentence
        (e.g., [['2.7.11.1'], ['3.4.21.-']]).
    """
    endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
    subscription_key = os.environ.get('AZURE_OPENAI_KEY', '')
    deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'o4-mini')
    api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

    if not subscription_key:
        raise ValueError("AZURE_OPENAI_KEY environment variable is not set")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    ec_list = []
    with tqdm(total=len(sentences), desc="Processing EC extraction", unit="sentence") as pbar:
        for sentence in sentences:
            try:
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a bioinformatics expert specializing in enzyme function annotation "
                                "and EC (Enzyme Commission) number assignment based on protein function descriptions."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Protein description:\n\"{sentence}\"\n\n"
                                "Task:\n"
                                "Determine whether the described protein has enzymatic activity, "
                                "and assign the most appropriate EC number(s) if applicable.\n\n"
                                "Instructions:\n"
                                "1. First, decide whether the protein is an enzyme.\n"
                                "   - If the description does NOT indicate catalytic activity, return exactly:\n"
                                "     ['-.-.-.-']\n"
                                "   - In this case, do NOT output any other EC numbers.\n"
                                "2. If the protein IS an enzyme, infer and assign EC number(s) based on the described catalytic activity, "
                                "even if no explicit 'EC x.x.x.x' pattern appears in the text.\n"
                                "3. When the protein is an enzyme, do NOT output ['-.-.-.-'].\n"
                                "4. Output EC numbers in canonical format a.b.c.d, where each level is either an integer or '-'.\n"
                                "5. If only partial EC information can be inferred, use '-' for unknown levels "
                                "(e.g., '2.7.-.-', '3.4.21.-').\n"
                                "6. Do NOT hallucinate EC numbers beyond what can be reasonably inferred from the description.\n"
                                "7. Output must be a Python-style list of strings, for example:\n"
                                "   ['3.5.1.13', '3.5.1.14', '3.5.1.-']\n"
                                "8. Do not include any explanation or additional text outside the list.\n"
                            ),
                        }
                    ],
                    max_tokens=1024,
                    temperature=0.3,
                    top_p=1.0,
                    model=deployment,
                )

                extracted_content = response.choices[0].message.content.strip()

                try:
                    import ast
                    ec_numbers = ast.literal_eval(extracted_content)
                    if not isinstance(ec_numbers, list):
                        ec_numbers = []
                    ec_numbers = [ec.replace('–', '-').replace('—', '-') for ec in ec_numbers if isinstance(ec, str)]
                except (ValueError, SyntaxError):
                    ec_numbers = re.findall(r'\b\d+\.[\d–-]+\.[\d–-]+\.[\d–-]+\b', extracted_content)
                    ec_numbers = [ec.replace('–', '-').replace('—', '-') for ec in ec_numbers]

                seen = set()
                ec_numbers = [x for x in ec_numbers if not (x in seen or seen.add(x))]

            except Exception as e:
                print(f"[Warning]: {e}")
                ec_numbers = []

            ec_list.append(ec_numbers)
            print('ec_numbers', ec_numbers)
            pbar.update(1)

    return ec_list
