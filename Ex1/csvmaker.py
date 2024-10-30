

def csv_maker(filename, data):
    with open(filename, 'w') as file:
        for row in data:
            file.write(','.join(map(str, row)) + "\n")
    print(f"File '{filename}' made sucsessfully")        
    
if __name__ == "__main__":
    datacsv = [
        [1,0,1,0,1,1,0],
        [1,0,0,1,1,1,1]
    ]
    
    csv_maker("data.csv", datacsv)