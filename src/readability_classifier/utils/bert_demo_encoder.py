from transformers import BertTokenizer

from src.readability_classifier.encoders.bert_encoder import BertEncoder

LIMIT = 100  # Maximum number of tokens to decode

if __name__ == "__main__":
    bert_encoder = BertEncoder()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Example code snippet
    code_snippet = """
/**
* This method determines the sign of a given number and prints a corresponding message.
*
* @param number The input number to be checked.
*/
public static void printSign(int number) {
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
    converted_lines: list[list[str]] = []
    for line in code:
        encoded_line = bert_encoder.encode_text(line)
        input_ids = encoded_line["input_ids"].tolist()[0]
        decoded_tokens = [tokenizer.decode(token_id) for token_id in input_ids]

        # Remove all whitespaces within a token
        decoded_tokens = [token.replace(" ", "") for token in decoded_tokens]

        # Remove all ## within a token
        decoded_tokens = [token.replace("##", "") for token in decoded_tokens]

        converted_lines.append(decoded_tokens)

    # Remove all [CLS], [SEP] and [PAD] tokens from the decoded code
    converted_lines = [
        [token for token in line if token not in ["[CLS]", "[SEP]", "[PAD]"]]
        for line in converted_lines
    ]
    converted_lines = [["[CLS]"]] + converted_lines

    # Remove tokens after the limit
    if LIMIT is not None and LIMIT > 0:
        limit = LIMIT - 1
        count = 0
        for i, line in enumerate(converted_lines):
            if count + len(line) > limit:
                converted_lines[i] = line[: limit - count]
                converted_lines = converted_lines[: i + 1]
                break
            count += len(line)

    converted_lines[-1].append("[SEP]")

    # Combine outer lists with "\n" and inner lists with " "
    decoded_code = "\n".join([" ".join(line) for line in converted_lines])

    print(decoded_code)
