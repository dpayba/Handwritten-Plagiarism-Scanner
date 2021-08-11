import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal

parser = argparse.ArgumentParser()
parser.add_argument('--file1_dir', help='Path to first comparison file', required=True)
parser.add_argument('--file2_dir', help='Path to second comparison file', required=True)
args = parser.parse_args()


file1 = args.file1_dir
file2 = args.file2_dir

parent_dir = os.path.dirname(file1)
file_names = os.listdir(parent_dir)

files_list = [file1, file2]
file_contents = [open(file1).read(), open(file2).read()]

vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(file_contents)
s_vectors = list(zip(files_list, vectors))

def check_plagiarism():
    plagiarism_results = set()
    global s_vectors
    for file, text_vector in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((file, text_vector))
        del new_vectors[current_index]
        for file_b, vector_b in new_vectors:
            similarity_score = similarity(text_vector, vector_b)[0][1]
            pair = sorted((file, file_b))
            score = (pair[0], pair[1], ('%.2f' %(similarity_score * 100)))
            plagiarism_results.add(score)
    return plagiarism_results

for data in check_plagiarism():
    print(file_names[0] + ' and ' + file_names[1] + ' are ' + data[2] + '% similar.')

