from tqdm import tqdm
import random

if __name__ == '__main__':
    name = "./all_data.csv"
    test = open("./test_data.csv", "w")
    train = open("./train_data.csv", "w")
    val = open("./val_data.csv", "w")
    r = random.random()
    random.seed(2021)
    with open(name, "r") as all:
        count = 0
        dataset = all.readlines()
        dataset = dataset[1:]
        random.shuffle(dataset)
        dataset_len = len(dataset)
        testLen = int(dataset_len * 0.1)
        valLen = int(dataset_len * 0.2)
        for data in tqdm(dataset):
            count += 1
            if count <= testLen:
                test.write(data)
            elif count <= valLen:
                val.write(data)
            else:
                train.write(data)
    test.close()
    val.close()
    train.close()
