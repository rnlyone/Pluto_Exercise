
def read_csv(filename):
    data=[]
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data.append([int(item) if item.isdigit() else item for item in row])
    
    return data

def ones_in_row_sum(data):
    counter = 0
    for row_index, row in enumerate(data):
        for value_index, value in enumerate(row):
            if value == 1:
                counter += 1
        print(f"Ones in row {row_index+1}: {counter}")
        counter = 0
        
def row_summation(data):
    sum = 0
    for row_index, row in enumerate(data):
        for value_index, value in enumerate(row):
            sum += value
        print(f"the Summation of line {row_index+1}: {sum}")
        sum = 0
        
def perfect_ones_in_column_sum(data):
    transposed_data = list(zip(*data))
    col_sum = 0
    for col_index, col in enumerate(transposed_data):
        if all(value == 1 for value in col):
            col_sum +=1

    print(f"The Summation of Perfect Ones Column is: {col_sum}")
    
def dimension_check(data):
    if isinstance(data, list):
        return 1 + dimension_check(data[0])
    return 0
    
def cosine_similiarity(data):
    top = 0; bottom_a = 0; bottom_b = 0
    
    transposed_data = list(zip(*data))
    print(dimension_check(csvdata), len(csvdata))
    if (dimension_check(transposed_data) and len(data) == 2):
        for row_index, row in enumerate(transposed_data):
            top += (row[0]*row[1])
            bottom_a += pow(row[0], 2)
            bottom_b += pow(row[1], 2)
        bottom = pow(bottom_a, 0.5) * pow(bottom_b, 0.5)
        cosinesim = top / bottom
        print(f"The Cosine Similiarity of this Data is: {cosinesim}")
    else:
        print("the dim or row is not 2")
        
def confusion_matrix(data):
    tp = tn = fp = fn = 0
    transposed_data = list(zip(*data))
    if (dimension_check(transposed_data) and len(data) == 2):
        for row in transposed_data:
            if row[0] and row[1] == 1:
                tp += 1
            elif row[0] and  row[1] == 0:
                tn += 1
            elif row[0] == 1 and row[1] == 0:
                fp += 1
            elif row[0] == 0 and row[1] == 1:
                fn += 1
    return tp, tn, fp, fn

def classification_report(data):
    tp, tn, fp, fn = confusion_matrix(data)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f"Accuracy: {(tn+tp)/(tn+fp+tp+fn)}")
    print(f"Precision: {tp/(tp+fp)}")
    print(f"Recall: {tp/(tp+fn)}")
    print(f"F1 Score: {2*(precision*recall)/(precision+recall)}")

if __name__ == "__main__":
    filename = "data.csv"
    
    csvdata = read_csv(filename)
    
    ones_in_row_sum(csvdata)
    
    print("\n")
    
    row_summation(csvdata)
    
    print('\n')
    
    perfect_ones_in_column_sum(csvdata)
    
    print('\n')
    
    cosine_similiarity(csvdata)
    
    print('\n')
    
    classification_report(csvdata)