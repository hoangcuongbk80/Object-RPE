import glob

f_test = open("test_data_list.txt", 'w')
f_train = open("train_data_list.txt", 'w')

test = ['0003', '0011', '0017', '0025', '0039']
val = ['0010','0018', '0026', '0027', '0028', '0029', '0030', '0031', '0032',]

for i in range(0, 40):
    seq = 10000 + i
    seq_str = str(seq)[1:]
    data_dir = 'data/' + seq_str + '/'
    label_path = '/media/aass/783de628-b7ff-4217-8c96-7f3764de70d9/Warehouse_Dataset/' + data_dir
    label_addrs = glob.glob(label_path + '*-label.png')
    if seq_str in val:
        continue
    if seq_str in test:
        for j in range(0, len(label_addrs)):
            img_index = 1000000 + j + 1
            img_index_str = str(img_index)[1:]
            img_dir = data_dir + img_index_str 
            f_test.write(img_dir)
            f_test.write('\n')
    else:
        for j in range(0, len(label_addrs)):
            img_index = 1000000 + j +1
            img_index_str = str(img_index)[1:]
            img_dir = data_dir + img_index_str 
            f_train.write(img_dir)
            f_train.write('\n')
