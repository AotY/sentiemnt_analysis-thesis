# @Author : bamtercelboo
# @Datetime : 2018/4/23 22:40
# @File : word_analogy.py
# @Last Modify Time : 2018/4/23 22:40
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  word_analogy.py
    FUNCTION : None
"""

import os
import sys
import struct
import numpy as np
from scipy.spatial.distance import cosine
import multiprocessing as mp
from optparse import OptionParser
from format import format

MAX_VECTORS = 200000  # This script takes a lot of RAM (>2GB for 200K vectors),
# if you want to use the full 3M embeddings then you probably need to insert the
# vectors into some kind of database
FLOAT_SIZE = 4  # 32bit float


class Eval(object):
    def __init__(self, ctg):
        self.ctg = ctg
        self.acc = 0
        self.rank = 0
        self.num = 0

    def update(self, rank):
        if rank == 1:
            self.acc += 1
        self.rank += rank
        self.num += 1

    def evaluate(self):
        acc = self.acc / self.num
        mean_rank = self.rank / self.num
        print('Category: {}\n'
              'Total count: {}\n'
              'Accuracy: {}\n'
              'Mean rank: {}\n'.format(self.ctg, self.num, acc, mean_rank))


class Analogy(object):
    def __init__(self, vector_file, analogy_file, names, binary):
        slef.names = names
        self.vector_file = vector_file
        self.analogy_file = analogy_file
        self.binary = binary
        self.vector_dict = {}

        assert os.path.isfile(vector_file), "{} is not a file.".format(vector_file)
        assert os.path.isfile(analogy_file), "{} is not a file.".format(analogy_file)

        self.read_vector(self.vector_file)

        if self.analogy_file is "":
            self.Word_Analogy(analogy="./Data/analogy.txt")
        else:
            self.Word_Analogy(analogy=self.analogy_file)

    def read_vector(self, path):
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

                    #  word = word.decode('utf-8')
                    word = word.decode('utf-8').rstrip().split()[0]
                    binary_vector = f.read(FLOAT_SIZE * embedding_dim)
                    self.vector_dict[word] = np.array([struct.unpack_from('f', binary_vector, i)[
                                                      0] for i in range(0, len(binary_vector), FLOAT_SIZE)])
                    #  print('vector_dict[word]: ', self.vector_dict[word])
                    index += 1

                    if index % 2000 == 0:
                        sys.stdout.write(
                            "\rHandling with the {} lines, all {} lines.".format(index + 1, num_vectors))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_split = line.strip().split(' ')
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
                    values = line.strip().split(' ')
                    if len(values) == 1 or len(values) == 2:
                        continue
                    if len(values) != int(embedding_dim) + 1:
                        print("\nWarning {} -line.".format(index + 1))
                        continue
                    self.vector_dict[values[0]] = np.array(list(map(float, values[1:])))
                    if index % 2000 == 0:
                        sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
                sys.stdout.write("\rHandling with the {} lines, all {} lines.".format(index + 1, all_lines))
            print("\nembedding words {}, embedding dim {}.".format(len(self.vector_dict), embedding_dim))

    def worker(self, analogy, target, queue):
        line_no = 0
        result = Eval(target)
        with open(analogy, 'r', encoding='utf-8') as fr:
            while True:
                line = fr.readline()
                line_no += 1

                if not line:
                    print('target {} not found'.format(target))
                    break

                if line[0] != ':':
                    continue

                topic = line.split()[1].split('-')[0]

                if topic == target:
                    print('target {} found. Beginning...'.format(target))
                    break

            while True:
                line = fr.readline()
                line_no += 1

                if not line or line[0] == ':':
                    print('target {} finished'.format(target))
                    break

                words = line.split()
                assert len(words) == 4

                if any([w not in self.vector_dict for w in words]):
                    #  print('something is wrong with the {}-th line'.format(line_no))
                    continue

                v1 = np.add(np.subtract(self.vector_dict[words[1]], self.vector_dict[words[0]]), self.vector_dict[words[2]])
                # v1 = self.vector_dict[words[1]] - self.vector_dict[words[0]] +
                # self.vector_dict[words[2]]
                v2 = self.vector_dict[words[-1]]

                v2_rank = 1
                v2_score = cosine(v1, v2)

                for w, v in self.vector_dict.items():
                    if w in words:
                        continue

                    score = cosine(v1, v)

                    if score < v2_score:
                        v2_rank += 1

                result.update(v2_rank)

        queue.put(result)

    def Word_Analogy(self, analogy):
        queue = mp.Queue()
        #  category = ['capital', 'city', 'family']
        category = ['capital', 'city']
        processes = [mp.Process(target=self.worker, args=(analogy, x, queue)) for x in category]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        results = [queue.get() for p in processes]
        for res in results:
            res.evaluate()

        acc, rank, num = 0, 0, 0
        for res in results:
            acc += res.acc
            rank += res.rank
            num += res.num
        print('Total acc: {}\nTotal mean rank: {}\n Total number: {}\n'.
              format(format(acc / num, spec=self.names), fromat(rank / num), format(num))

if __name__ == "__main__":
    print("Word Analogy Evaluation")
    # vector_file = "./Data/zhwiki_substoke.100d.source"
    # analogy_file = "./Data/analogy.txt"
    # Analogy(vector_file=vector_file, analogy_file=analogy_file)

    parser = OptionParser()
    parser.add_option("--vector", dest="vector", help="vector file")
    parser.add_option("--analogy", dest="analogy", default="", help="analogy file")
    parser.add_option("--binary", dest="binary", action='store_true', help="similarity file")
    (options, args) = parser.parse_args()

    vector_file = options.vector
    analogy_file = options.analogy
    binary = options.binary
    names = []
    names.append(similarity_file.split('/')[-1][:-4])
    names.extend(vector_file.split('merge.')[-1].split('.')[:3])

    try:
        Analogy(vector_file, analogy_file, names, binary=binary)
        print("All Finished.")
    except Exception as err:
        print(err)


