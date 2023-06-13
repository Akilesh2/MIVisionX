import sys
golden_file_path=sys.argv[1]
rocal_file_path=sys.argv[2]
# golden_file = open(golden_file_path, 'r')
# rocal_file = open(rocal_file_path, 'r')
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
            # else:
            #     print(f"Lines do not match: {line1} | {line2}")
        print("expected values ",lst1)
        print("obtained values ",lst2)

    # Check if one file has more lines than the other
    # if len(list(file1)) != len(list(file2)):
    #     print("Files have different number of lines.")

# Usage example
compare_files(golden_file_path, rocal_file_path)
