import os

def main():
    os.system('python controller.py --img_file ../data/line.png --output_dir ../text_output/file1.txt')
    os.system('python controller.py --img_file ../data/line.png --output_dir ../text_output/file2.txt')

if __name__ == '__main__':
    main()