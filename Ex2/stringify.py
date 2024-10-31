import re

def down_marker(string):
    pattern = r"<.*?>"
    
    cleaned_text = re.sub(pattern, '', string)
    
    return cleaned_text

def tokenizer(string):
    pattern = r">[^<]+</"
    
    extracted = re.findall(pattern, string)
    
    cleaned_extraction = [sentences[1:-2].strip() for sentences in extracted]

    result_array = []

    for text in cleaned_extraction:
        result_array.append(text)
    
    return result_array

def exp_counter(arr):
    return len(arr)
    


if __name__ == "__main__":
    string = "<tag1>tag1 data</tag1> no tag no data <tag1><tag2>tag2 data</tag2></tag1>"
    print(f"Markdown Text: '{down_marker(string)}'")
    
    print("\n")
    
    print(f"Tokenized Text: {tokenizer(string)}")
    
    print("\n")
    
    print(f"Expression Count: {exp_counter(tokenizer(string))}")    
    
