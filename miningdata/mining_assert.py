import os 
with open('./new_minging.txt','r') as file:
    with open('./asserted_minging.txt','w') as f:
        for line in file:
            address=line.strip().split()[0]
            if os.path.exists(address):
                print(line,file=f,end='')

