# ===============================================================
# src/attack/b4/prompts.py
# Description: Contains a collection of prompt templates for the
#              paraphrasing task in the B4 attack.
# ===============================================================

# List of various prompt templates for paraphrasing
id2prompt = [
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice. This is the text: \n{}",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases.\n{}",
    "Paraphrase the following paragraphs line by line. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "Paraphrase the following paragraphs line by line. Try to keep the similar length to the original paragraphs. Don't output any other information except the paraphrased texts.\nThis is the text:\n{}",
    "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Paraphrase all sentences one by one. Don't output any other information except the paraphrased texts. This is the text:\n{}",
    "Paraphrase the following paragraph such that it preserves the original meaning but uses different phrasing and vocabulary. Ensure that the new version has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Output should be natural, coherent, and maintain the key information from the source text. Here are the texts:\n{}",
    "Paraphrase the following paragraph line by line, such that it preserves the original meaning but uses different phrasing and vocabulary. Ensure that the new version has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here are the texts:\n{}",
    "Paraphrase the following paragraph such that it preserves the original meaning but has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here is the text:\n{}",
    "Paraphrase the following paragraph in your own tone. Ensure that it has minimal overlap with the original in terms of common phrases, word sequences, and n-grams. Here is the texts:\n{}",
    "Rewrite the following paragraph in a way that retains its core meaning but alters its wording and structure. Focus on minimizing shared n-grams and phrases between the original and the rewritten text, while keeping the content clear and coherent. Here are the texts:\n{}",
    "Transform the following paragraph into a new version that conveys the same message but is expressed with different wording and phrasing. Try to keep n-gram overlaps minimal, employing synonyms, rephrased expressions, and varied sentence patterns. Here are the texts:\n{}",
    "Create a paraphrased version of the provided text such that it maintains the semantic essence while minimizing the similarity in wording and n-gram patterns. Focus on using distinct phrases and vocabulary to achieve a high degree of linguistic diversity. Here are the texts:\n{}",
]

def get_prompt(args):
    """
    Selects a prompt template based on the provided configuration or ID.

    Args:
        args: Can be an integer ID or a configuration object with a 'prompt_id' attribute.

    Returns:
        A formatted string template for the prompt.
    """
    if isinstance(args, int):
        prompt_id = args
    else:
        # Assumes args is a config object with a 'prompt_id' attribute
        prompt_id = args.prompt_id
    
    if 0 <= prompt_id < len(id2prompt):
        return id2prompt[prompt_id]
    else:
        # Fallback to the first prompt if the ID is invalid
        return id2prompt[0]
