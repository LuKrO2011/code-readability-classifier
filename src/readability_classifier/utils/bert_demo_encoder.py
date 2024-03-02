from transformers import BertTokenizer

from readability_classifier.encoders.bert_encoder import BertEncoder

LIMIT = 100  # Maximum number of tokens to decode

if __name__ == "__main__":
    bert_encoder = BertEncoder()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Example code snippet
    code_snippet = """
    /**
    * This method determines the sign of a given number and prints a corresponding.
    *
    * @param number The input number to be checked.
    */
    public static void printSignTest123ShouldBeSplit(int number) {
        if (number > 0) {
            System.out.println("Number is positive");
        } else if (number < 0) {
            System.out.println("Number is negative");
        } else {
            System.out.println("Number is zero");
        }
    }
    """

    # Split the code snippet into lines
    code = code_snippet.split("\n")

    # Encode and decode each line of the code snippet
    converted_lines = []
    for line in code:
        encoded_line = bert_encoder.encode_text(line)
        input_ids = encoded_line["input_ids"].tolist()[0]
        decoded_tokens = [tokenizer.decode(token_id) for token_id in input_ids]

        # Remove all whitespaces within a token
        decoded_tokens = [token.replace(" ", "") for token in decoded_tokens]

        # Remove all ## within a token
        decoded_tokens = [token.replace("##", "") for token in decoded_tokens]

        converted_lines.append(decoded_tokens)

    # Join the decoded lines back into a code snippet
    decoded_code = "\n".join(converted_lines)

    # Remove all [CLS], [SEP] and [PAD] tokens from the decoded code
    decoded_code = (
        decoded_code.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
    )

    # Add [CLS] and [SEP] tokens to the decoded code
    decoded_code = f"[CLS]\n{decoded_code}\n[SEP]"

    # Remove tokens tha
    if LIMIT is not None and LIMIT > 0:
        converted_lines = [line[:LIMIT] for line in converted_lines]

    print(decoded_code)
