import sys
import os

def compare_files(file1_path, file2_path):
    lines1,lines2,lst1,lst2 = [],[],[],[]
    print("check in compare.py ")
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            line1 = line1.strip()
            line2 = line2.strip()
            # print("hello")
            if line1 != line2:
                print(f"Lines not match: {line1}")
                lst1.append(line1)
                lst2.append(line2)
            break
        print("expected values ",lst1)
        print("obtained values ",lst2)


def main():
    print("check")
    ref_output_path = sys.argv[1]
    rocal_output_path = sys.argv[2]
    if not (os.path.exists(ref_output_path) and os.path.exists(rocal_output_path)):
        logging.error("Path does not Exists")
        exit()
    golden_output_dir_list = os.listdir(ref_output_path)
    rocal_output_dir_list = os.listdir(rocal_output_path)
    golden_file_path = ""
    for aug_name in rocal_output_dir_list:
        rocal_file_path=rocal_output_path+aug_name
        temp = aug_name.split('.')
        file_name_s = temp[0].split('_')
        if "labels" in file_name_s:
            golden_file_path=ref_output_path+"/labels_golden.txt"
        elif "caffe" in file_name_s:
            golden_file_path=ref_output_path+"/caffe_bbox_golden.txt"
        elif "tf" in file_name_s:
            golden_file_path=ref_output_path+"/tf_bbox_golden.txt"
        elif "coco" in file_name_s:
            golden_file_path=ref_output_path+"/coco_bbox_golden.txt"
        print("golden_file_path ",golden_file_path)
        print("rocal_file_path  ",rocal_file_path)
        compare_files(golden_file_path, rocal_file_path)

if __name__ == '__main__':
    main()

        
        


