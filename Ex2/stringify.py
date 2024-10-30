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
    
def LIS_sagashi(S):
    L = []
    for x in S:
        if len(L) == 0 or x > L[-1]:
            L.append(x)
        else:
            pos = binary_search(L, x)
            L[pos] = x
    return len(L)

def binary_search(L, x):
    left, right = 0, len(L) - 1
    while left < right:
        mid = (left + right) // 2
        if L[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

def binary_search(L, x):
    left, right = 0, len(L) - 1
    while left < right:
        mid = (left + right) // 2
        if L[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left


def LCS(S, T):
    if len(S) < len(T):
        S, T = T, S
    
    previous = [0] * (len(T) + 1)
    current = [0] * (len(T) + 1)
    
    for i in range(1, len(S) + 1):
        for j in range(1, len(T) + 1):
            if S[i - 1] == T[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        
        previous, current = current, previous  
        
    return previous[len(T)]
