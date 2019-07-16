# @Author : bamtercelboo
# @Datetime : 2018/4/23 19:00
# @File : word_similarity.py
# @Last Modify Time : 2018/4/23 19:00
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  word_similarity.py
    FUNCTION : None
"""

import os
import sys
import struct
import logging
import numpy as np
from optparse import OptionParser
from scipy import linalg

MAX_VECTORS = 200000  # This script takes a lot of RAM (>2GB for 200K vectors),
# if you want to use the full 3M embeddings then you probably need to insert the
# vectors into some kind of database
FLOAT_SIZE = 4  # 32bit float


class Similarity(object):
    def __init__(self, vector_file, similarity_file, binary=False):
        self.binary = binary
        self.vector_dict = {}
        self.result = {}
        self.vector_file = vector_file
        self.similarity_file = similarity_file
        if self.similarity_file is "":
            self.Word_Similarity(
                similarity_name="./Data/wordsim-240.txt", vec=self.vector_dict)
            #  self.Word_Similarity(similarity_name="./Data/wordsim-297.txt")
        else:
            self.Word_Similarity(similarity_name=self.similarity_file)

        self.read_vector(self.vector_file)
        self.pprint(self.result)

    def cos(self, vec1, vec2):
        return vec1.dot(vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    def read_vector(self, path):
        print('read vector ... %s' % path)
        assert os.path.isfile(path), "{} is not a file.".format(path)
        embedding_dim = -1
        if self.binary:
            with open(self.vector_file, 'rb') as f:
                c = None
                # read the header
                header = b""
                while c != b"\n":
                    c = f.read(1)
                    header += c
                #  print('header: ', header)

                num_vectors, embedding_dim = (int(x) for x in header.split())
                #  num_vectors = min(MAX_VECTORS, total_num_vectors)

                print("Number of vectors: %d" % (num_vectors))
                print("Vector size: %d" % embedding_dim)

                index = 0
                while len(self.vector_dict) < num_vectors:
                    word = b""
                    while True:
                        c = f.read(1)
                        if c == b" ":
                            break
                        word += c

                    word = word.decode('utf-8').rstrip().split()[0]
                    binary_vector = f.read(FLOAT_SIZE * embedding_dim)
                    self.vector_dict[word] = np.array([struct.unpack_from('f', binary_vector, i)[
                                                      0] for i in range(0, len(binary_vector), FLOAT_SIZE)])
                    #  print('word: {} vec: {}'.format(word, self.vector_dict[word]))
                    #  print('word: {} vec: {}'.format(word, self.vector_dict[word]))
                    #  print('vector_dict[{}]: {} '.format(word,  self.vector_dict[word]))
                    index += 1

                    if index % 2000 == 0:
                        sys.stdout.write(
                            "\rHandling with the {} lines, all {} lines.".format(index + 1, num_vectors))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_split = line.strip().split()
                    if len(line_split) == 1:
                        embedding_dim = line_split[0]
                        break
                    elif len(line_split) == 2:
                        embedding_dim = line_split[1]
                        break
                    else:
                        embedding_dim = len(line_split) - 1
                        break

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_lines = len(lines)
                index = 0
                for index, line in enumerate(lines):
                    values = line.strip().split()
                    if len(values) == 1 or len(values) == 2:
                        continue

                    if len(values) != int(embedding_dim) + 1:
                        # print("Warning {} -line.".format(index + 1))
                        logging.info("Warning {} -line.".format(index + 1))
                        continue

                    self.vector_dict[values[0]] = np.array(
                        list(map(float, values[1:])))

                    if index % 2000 == 0:
                        sys.stdout.write(
                            "\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
                sys.stdout.write(
                    "\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
        print("\nembedding words {}, embedding dim {}.".format(
            len(self.vector_dict), embedding_dim))

    def pprint(self, result):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k, v in result.items():
            x.add_row([k, v[0], v[1], v[2]])
        print(x)

    def Word_Similarity(self, similarity_name):
        pred, label, found, notfound = [], [], 0, 0
        with open(similarity_name, encoding='utf-8') as fr:
            for i, line in enumerate(fr):
                w1, w2, score = line.split()
                #  print('w1: %s w2: %s' % (w1, w2))
                if w1 in self.vector_dict and w2 in self.vector_dict:
                    found += 1
                    pred.append(
                        self.cos(self.vector_dict[w1], self.vector_dict[w2]))
                    label.append(float(score))
                else:
                    notfound += 1
        file_name = similarity_name[similarity_name.rfind(
            "/") + 1:].replace(".txt", "")
        self.result[file_name] = (found, notfound, self.cos(label, pred))


if __name__ == "__main__":
    print("Word Similarity Evaluation")

    # vector_file = "./Data/zhwiki_substoke.100d.source"
    # vector_file = "./Data/zhwiki_cbow.100d.source"
    # similarity_file = "./Data/wordsim-297.txt"
    # Similarity(vector_file=vector_file, similarity_file=similarity_file)
    # Similarity(vector_file=vector_file, similarity_file="")

    parser = OptionParser()
    parser.add_option("--vector", dest="vector", help="vector file")
    parser.add_option("--similarity", dest="similarity",
                      default="", help="similarity file")
    parser.add_option("--binary", dest="binary",
                      action='store_true', help="similarity file")
    (options, args) = parser.parse_args()

    vector_file = options.vector
    similarity_file = options.similarity
    binary = options.binary

    try:
        Similarity(vector_file, similarity_file, binary=binary)
        print("All Finished.")
    except Exception as err:
        print(err)

    # stats.stats.spearmanr(vec1, vec2)[0]
