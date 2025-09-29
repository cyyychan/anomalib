import sys
import os
import glob


if __name__ == "__main__":
    dir_path = sys.argv[1]
    train_list = glob.glob(os.path.join(dir_path, "train/**/*.*"), recursive=True)
    test_list = glob.glob(os.path.join(dir_path, "test/**/*.*"), recursive=True)

    train_txt_path = os.path.join(dir_path, "train.txt")
    with open(train_txt_path, "w") as fs:
        for img_path in train_list:
            label = img_path.split("/")[-2]
            fs.write(img_path+","+label+"\n")

    test_txt_path = os.path.join(dir_path, "test.txt")
    with open(test_txt_path, "w") as fs:
        for img_path in test_list:
            label = img_path.split("/")[-2]
            fs.write(img_path+","+label+"\n")