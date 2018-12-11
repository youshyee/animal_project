import os
orig_name = ''
orig_num = 0
with open('new_mineddata.txt', 'w') as file:
    with open('./mineddata.txt', 'r') as f:
        for line in f:
            line = line.strip()
            line_list = line.split()
            address = line_list[0]
            data1 = line_list[1]
            key_addr = address.split('/')[-1]
            key_frame_name = key_addr.split('_')[0]
            num = key_addr.split('_')[-1].split('.')[0]
            num = int(num)
            cls = data1.split(',')[-1]
            cls = int(cls)
            if cls == 0:
                if orig_name != key_frame_name:
                    print(line, file=file)
                    orig_name = key_frame_name
                    orig_num = num
                    print(num)
                else:
                    if num >= orig_num+5:
                        print(line, file=file)
                        orig_num = num
            else:
                print(line, file=file)
