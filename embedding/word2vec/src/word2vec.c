//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define WORD_TYPE 110
#define PINYIN_TYPE 111

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int pinyin_vocab_hash_size = 20000000;

typedef float real;                    // Precision of float numbers

struct VocabWord {
	long long count;
	long long pinyin_idx; //
	/* char *pinyin;  */
	int *point;
	char *word, *code, codelen;
};

char train_word_file[MAX_STRING], output_file[MAX_STRING];
char train_pinyin_file[MAX_STRING];

char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char save_pinyin_vocab_file[MAX_STRING], read_pinyin_vocab_file[MAX_STRING];

struct VocabWord *vocab;

int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;

long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;

long long pinyin_vocab_max_size = 1000, pinyin_vocab_size = 0;
int *pinyin_vocab_hash;
/* char **pinyin_vocab; // vocab for pinyin */
struct VocabPinyin {
	long long count;
	char *pinyin;
};
struct VocabPinyin *pinyin_vocab;
int model_type = 1; //  1: original; 2: joint Pinyin; 3: TF-IDF weights-sum; 4: pinyin + TF-IDF, default 1

real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *exp_table;
real *pinyin_v; // pinyin vector

clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void initUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	if (table == NULL) {
		fprintf(stderr, "cannot allocate memory for the table\n");
		exit(1);
	}
	for (a = 0; a < vocab_size; a++)
		train_words_pow += pow(vocab[a].count, power);
	i = 0;
	d1 = pow(vocab[i].count, power) / (real)train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].count, power) / (real)train_words_pow;
		}
		if (i >= vocab_size)
			i = vocab_size - 1;
	}
}

// Reads a single word or pinyin from a file, assuming space + tab + EOL to be word boundaries
void readStr(char *str, FILE *fin) {
	int len = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13)  // CR  is a bytecode for carriage return
			continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (len > 0) {
				if (ch == '\n')
					ungetc(ch, fin);
				break;
			}
			if (ch == '\n') { // len == 0
				strcpy(str, (char *)"</s>");
				return;
			} else
				continue;
		}
		str[len] = ch;
		len++;
		if (len >= MAX_STRING - 1)
			len--;   // Truncate too long words
	}
	str[len] = 0;
}

// Returns hash value of a word (or pinyin)
int getStrHash(char *str, int type) {
	unsigned long long i, hash = 0;
	for (i = 0; i < strlen(str); i++)
		hash = hash * 257 + str[i];
	if (type == WORD_TYPE)
		hash = hash % vocab_hash_size;
	else
		hash = hash % pinyin_vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int searchWord(char *word) {
	unsigned int hash = getStrHash(word, WORD_TYPE);
	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word))
			return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

int searchPinyin(char *pinyin) {
	unsigned int hash = getStrHash(pinyin, PINYIN_TYPE);
	while (1) {
		if (pinyin_vocab_hash[hash] == -1)
			return -1;
		hash = (hash + 1) % pinyin_vocab_hash_size;
	}
	return -1;
}
// Reads a word and returns its index in the vocabulary
// type: WORD - word, PINYIN - pinyin
int readWordIndex(FILE *fin) {
	char str[MAX_STRING];
	readStr(str, fin);
	if (feof(fin))
		return -1;
	return searchWord(str);
}

// Adds a word to the vocabulary, return cur index
int addWordToVocab(char *word, int pinyin_idx) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING)
		length = MAX_STRING;

	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);

	if (model_type == 2 || model_type == 4)
		vocab[vocab_size].pinyin_idx = pinyin_idx;

	vocab[vocab_size].count = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct VocabWord *)realloc(vocab, vocab_max_size * sizeof(struct VocabWord));
	}
	hash = getStrHash(word, WORD_TYPE);
	while (vocab_hash[hash] != -1)
		hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// add a pinyin to pinyin vocabulary, return cur index
int addPinyinToVocab(char *pinyin) {
	unsigned int hash, length = strlen(pinyin) + 1;
	if (length > MAX_STRING)
		length = MAX_STRING;

	pinyin_vocab[pinyin_vocab_size].pinyin = (char *)calloc(length, sizeof(char));
	strcpy(pinyin_vocab[pinyin_vocab_size].pinyin, pinyin);

	pinyin_vocab[pinyin_vocab_size].count = 0;

	pinyin_vocab_size++;
	// Reallocate memory if needed
	if (pinyin_vocab_size + 2 >= pinyin_vocab_max_size) {
		pinyin_vocab_max_size += 1000;
		/* pinyin_vocab = (char **)realloc(pinyin_vocab, pinyin_vocab_max_size * sizeof(char *)); */
		pinyin_vocab = (struct VocabPinyin *)realloc(pinyin_vocab, pinyin_vocab_max_size * sizeof(struct VocabPinyin));
	}

	hash = getStrHash(pinyin, PINYIN_TYPE);
	while (pinyin_vocab_hash[hash] != -1)
		hash = (hash + 1) % pinyin_vocab_hash_size;

	pinyin_vocab_hash[hash] = pinyin_vocab_size - 1;
	return pinyin_vocab_size - 1;
}

// Used later for sorting by word counts
int vocabCompare(const void *a, const void *b) {
	return ((struct VocabWord *)b)->count - ((struct VocabWord *)a)->count;
}

int pinyinVocabCompare(const void *a, const void *b) {
	return ((struct VocabPinyin *)b)->count - ((struct VocabPinyin *)a)->count;
}

// Sorts the vocabulary by frequency using word counts
void sortVocab() {
	printf("sortVocab...\n");
	long long i, size;
	unsigned int hash;

	// sort pinyin vocab
	if (model_type == 2 || model_type == 4) {
		qsort(&pinyin_vocab[1], pinyin_vocab_size - 1, sizeof(struct VocabPinyin), pinyinVocabCompare);
		for (i = 0; i < pinyin_vocab_hash_size; i++)
			pinyin_vocab_hash[i] = -1;

		for (i = 0; i < pinyin_vocab_size; i++) { // Skip </s>
			hash = getStrHash(pinyin_vocab[i].pinyin, PINYIN_TYPE);
			while (pinyin_vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;
			pinyin_vocab_hash[hash] = i;
		}
	}

	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct VocabWord), vocabCompare);
	for (i = 0; i < vocab_hash_size; i++)
		vocab_hash[i] = -1;

	size = vocab_size;
	train_words = 0;
	for (i = 1; i < size; i++) { // Skip </s>
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[i].count < min_count) {
			vocab_size--;
			free(vocab[i].word);
			vocab[i].word = NULL;
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash = getStrHash(vocab[i].word, WORD_TYPE);
			while (vocab_hash[hash] != -1)
				hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = i;
			train_words += vocab[i].count;
		}
	}
	vocab = (struct VocabWord *)realloc(vocab, (vocab_size + 1) * sizeof(struct VocabWord));

	// Allocate memory for the binary tree construction
	for (i = 0; i < vocab_size; i++) {
		vocab[i].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[i].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void readuceVocab() {
	printf("readuceVocab...\n");
	int i, j = 0;
	unsigned int hash;
	for (i = 0; i < vocab_size; i++){
		if (vocab[i].count > min_reduce) {
			vocab[j].count = vocab[i].count;
			vocab[j].word = vocab[i].word;
			if (model_type == 2 || model_type == 4)
				vocab[j].pinyin_idx = vocab[i].pinyin_idx;
			j++;
		} else
			free(vocab[i].word);
	}
	vocab_size = j;
	// re init vocab_hash
	for (i = 0; i < vocab_hash_size; i++)
		vocab_hash[i] = -1;

	for (i = 0; i < vocab_size; i++) {
		// Hash will be re-computed, as it is not actual
		hash = getStrHash(vocab[i].word, WORD_TYPE);
		while (vocab_hash[hash] != -1)
			hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = i;
	}
	fflush(stdout);
	min_reduce++;
}

void learnVocabFromTrainFile() {
	char word[MAX_STRING];
	char pinyin[MAX_STRING];
	FILE *fin_word;
	FILE *fin_pinyin;
	long long i, word_idx, pinyin_idx = 0;
	for (i = 0; i < vocab_hash_size; i++)
		vocab_hash[i] = -1;

	fin_word = fopen(train_word_file, "rb");
	if (fin_word == NULL) {
		printf("ERROR: training word data file not found!\n");
		exit(1);
	}

	if (model_type == 2 || model_type == 4) {
		for (i = 0; i < pinyin_vocab_hash_size; i++)
			pinyin_vocab_hash[i] = -1;

		fin_pinyin = fopen(train_pinyin_file, "rb");
		if (fin_pinyin == NULL) {
			printf("ERROR: training pinyin data file not found!\n");
			exit(1);
		}
		pinyin_vocab_size = 0;
		addPinyinToVocab((char *)"none"); // none means word has not pinyin
	}

	vocab_size = 0;
	addWordToVocab((char *)"</s>", 0); // for \n
	while (1) {
		readStr(word, fin_word);
		if (feof(fin_word))
			break;
		if (model_type == 2 || model_type == 4) {
			readStr(pinyin, fin_pinyin);
			if (feof(fin_pinyin))
				break;
		}

		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		// check pinyin is exists ?
		if (model_type == 2 || model_type == 4) {
			pinyin_idx = searchPinyin(pinyin);
			if (pinyin_idx == -1) {
				pinyin_idx = addPinyinToVocab(word);
				pinyin_vocab[pinyin_idx].count = 1;
			} else
				pinyin_vocab[pinyin_idx].count++;
		}

		word_idx = searchWord(word); // if word is exists.
		if (word_idx == -1) {
			word_idx = addWordToVocab(word, pinyin_idx);
			vocab[word_idx].count = 1;
		} else
			vocab[word_idx].count++;

		if (vocab_size > vocab_hash_size * 0.7)
			readuceVocab();
	}
	sortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Max vocab size: %lld\n", vocab_max_size);
		printf("Vocab[0].word: %s\n", vocab[0].word);
		printf("Vocab[0].count: %lld\n", vocab[0].count);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin_word);
	fclose(fin_word);

	if (fin_pinyin != NULL){
		file_size = ftell(fin_pinyin);
		fclose(fin_pinyin);
	}
}

void destroyVocab() {
	int i;

	for (i = 0; i < vocab_size; i++) {
		if (vocab[i].word != NULL) {
			free(vocab[i].word);
		}
		if (vocab[i].code != NULL) {
			free(vocab[i].code);
		}
		if (vocab[i].point != NULL) {
			free(vocab[i].point);
		}
	}
	free(vocab[vocab_size].word);
	free(vocab);

	for (i = 0; i < pinyin_vocab_size; i++) {
		if (pinyin_vocab[i].pinyin != NULL)
			free(pinyin_vocab[i].pinyin);
	}
	free(pinyin_vocab[pinyin_vocab_size].pinyin);
	free(pinyin_vocab);
}



// Create binary Huffman tree using the word counts
// Frequent words will have short unique binary codes
void createBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];

	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));

	for (i = 0; i < vocab_size; i++)
		count[i] = vocab[i].count;

	for (i = vocab_size; i < vocab_size * 2; i++)
		count[i] = 1e15;

	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (i = 0; i < vocab_size - 1; i++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + i] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + i;
		parent_node[min2i] = vocab_size + i;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (i = 0; i < vocab_size; i++) {
		a = i;
		b = 0;
		while (1) {
			code[b] = binary[a];
			point[b] = a;
			b++;
			a = parent_node[a];
			if (a == vocab_size * 2 - 2)
				break;
		}
		vocab[i].codelen = b;
		vocab[i].point[0] = vocab_size - 2;
		for (a = 0; a < b; a++) {
			vocab[i].code[b - a - 1] = code[a];
			vocab[i].point[b - a] = point[a] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

void saveVocab(int type) {
	long long i;
	FILE *fo;
	if (type == WORD_TYPE){
		fo = fopen(save_vocab_file, "wb");
		for (i = 0; i < vocab_size; i++)
			fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].count);
	} else{
		fo = fopen(save_pinyin_vocab_file, "wb");
		for (i = 0; i < pinyin_vocab_size; i++)
			fprintf(fo, "%s %lld\n", pinyin_vocab[i].pinyin, pinyin_vocab[i].count);
	}
	fclose(fo);
}

void readVocab() {
	long long a, i = 0;
	char c;
	FILE *fin_word = fopen(read_vocab_file, "rb");
	FILE *fin_pinyin;
	char word[MAX_STRING];
	if (fin_word == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (i = 0; i < vocab_hash_size; i++)
		vocab_hash[i] = -1;

	while (1) {
		readStr(word, fin_word);
		if (feof(fin_word))
			break;
		a = addWordToVocab(word, 0);
		fscanf(fin_word, "%lld%c", &vocab[a].count, &c);
		i++;
	}

	if (model_type == 2 || model_type == 4) {
		fin_pinyin = fopen(read_pinyin_vocab_file, "rb");
		if (fin_pinyin == NULL) {
			printf("Vocabulary pinyin file not found\n");
			exit(1);
		}
		char pinyin[MAX_STRING];
		pinyin_vocab_size = 0;
		while (1) {
			readStr(pinyin, fin_pinyin);
			if (feof(fin_pinyin))
				break;
			a = addPinyinToVocab(pinyin);
			fscanf(fin_pinyin, "%lld%c", &pinyin_vocab[a].count, &c);
			i++;
		}
		fclose(fin_pinyin);

	}

	sortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}

	fin_word = fopen(train_word_file, "rb");
	if (fin_word == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin_word, 0, SEEK_END);
	file_size = ftell(fin_word);
	fclose(fin_word);

	if (fin_pinyin != NULL){
		file_size = ftell(fin_pinyin);
		fclose(fin_pinyin);
	}
}

void initNet() {
	printf("Init Net...");
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				syn1[a * layer1_size + b] = 0;
	}
	if (negative > 0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < vocab_size; a++)
				syn1neg[a * layer1_size + b] = 0;
	}
	for (b = 0; b < layer1_size; b++)
		for (a = 0; a < vocab_size; a++)
			syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;

	if (model_type == 2 || model_type == 4) {
		a = posix_memalign((void **)&pinyin_v, 128, (long long)pinyin_vocab_size * layer1_size * sizeof(real));
		if (pinyin_v == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < layer1_size; b++)
			for (a = 0; a < pinyin_vocab_size; a++)
				pinyin_v[a * layer1_size + b] = 0;
	}

	createBinaryTree();
}

void destroyNet() {
	if (syn0 != NULL) {
		free(syn0);
	}
	if (syn1 != NULL) {
		free(syn1);
	}
	if (syn1neg != NULL) {
		free(syn1neg);
	}

	if (pinyin_v != NULL) {
		free(pinyin_v);
	}
}

void *trainModelThread(void *id) {
	/* printf("trainModelThread %lld...", (long long)id); */
	/* fflush(stdout); */
	long long a, b, d, word, last_word, sentence_length = 0, sentence_pos = 0;
	long long word_count = 0, last_word_count = 0, sentence[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label;
	long long l1_pinyin;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	FILE *fi = fopen(train_word_file, "rb");
	if (fi == NULL) {
		fprintf(stderr, "no such file or directory: %s", train_word_file);
		exit(1);
	}
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	while (1) {
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now=clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
						alpha,
						word_count_actual / (real)(train_words + 1) * 100,
						word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
			if (alpha < starting_alpha * 0.0001)
				alpha = starting_alpha * 0.0001;
		}

		if (sentence_length == 0) {
			while (1) {
				word = readWordIndex(fi);
				if (feof(fi))
					break;
				if (word == -1)
					continue;
				word_count++;
				if (word == 0)
					break;

				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].count / (sample * train_words)) + 1) * \
							   (sample * train_words) / vocab[word].count;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536)
						continue;
				}
				sentence[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH)
					break;
			}
			sentence_pos = 0;
		}

		if (feof(fi))
			break;
		if (word_count > train_words / num_threads)
			break;

		word = sentence[sentence_pos];

		if (word == -1)
			continue;

		for (c = 0; c < layer1_size; c++)
			neu1[c] = 0;

		for (c = 0; c < layer1_size; c++)
			neu1e[c] = 0;

		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;

		if (cbow) {  //train the cbow architecture
			// in -> hidden
			for (a = b; a < window * 2 + 1 - b; a++) {
				if (a != window) {
					c = sentence_pos - window + a;
					if (c < 0 || c >= sentence_length)
						continue;

					last_word = sentence[c];
					if (last_word == -1)
						continue;

					// input: mean
					for (c = 0; c < layer1_size; c++)
						neu1[c] += syn0[c + last_word * layer1_size];

					// joint pinyin info
					if (model_type == 2 || model_type == 4) {
						long long pinyin_idx = vocab[last_word].pinyin_idx;
						for (c = 0; c < layer1_size; c++)
							neu1[c] += pinyin_v[c + pinyin_idx * layer1_size];
					}
				}
			}

			if (hs) {
				for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;

					l2 = vocab[word].point[d] * layer1_size;

					// output: Propagate hidden -> output
					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1[c + l2];

					if (f <= -MAX_EXP)
						continue;
					else if (f >= MAX_EXP)
						continue;
					else
						f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;

					// Propagate errors output -> hidden for input update
					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1[c + l2];

					// update: Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++)
						syn1[c + l2] += g * neu1[c];
				}
			}

			// NEGATIVE SAMPLING
			if (negative > 0) {
				for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];

						if (target == 0)
							target = next_random % (vocab_size - 1) + 1;
						if (target == word)
							continue;
						label = 0;
					}

					l2 = target * layer1_size;
					f = 0;

					for (c = 0; c < layer1_size; c++)
						f += neu1[c] * syn1neg[c + l2];

					if (f > MAX_EXP)
						g = (label - 1) * alpha;
					else if (f < -MAX_EXP)
						g = (label - 0) * alpha;
					else
						g = (label - exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

					// Propagate errors output -> hidden for input update
					for (c = 0; c < layer1_size; c++)
						neu1e[c] += g * syn1neg[c + l2];

					// update: Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++)
						syn1neg[c + l2] += g * neu1[c];
				}
			}

			// hidden -> in
			for (a = b; a < window * 2 + 1 - b; a++) {
				if (a != window) {
					c = sentence_pos - window + a;
					if (c < 0 || c >= sentence_length)
						continue;

					last_word = sentence[c];
					if (last_word == -1)
						continue;

					// update hidden -> input
					for (c = 0; c < layer1_size; c++)
						syn0[c + last_word * layer1_size] += neu1e[c];

					// update pinyin hidden -> input
					if (model_type == 2 || model_type == 4){
						long long pinyin_idx = vocab[last_word].pinyin_idx;
						for (c = 0; c < layer1_size; c++)
							pinyin_v[c + pinyin_idx * layer1_size] += neu1e[c];
					}
				}
			}
		} else {  //train skip-gram
			for (a = b; a < window * 2 + 1 - b; a++)
				if (a != window) {
					c = sentence_pos - window + a;
					if (c < 0 || c >= sentence_length)
						continue;

					last_word = sentence[c];
					if (last_word == -1)
						continue;

					l1 = last_word * layer1_size;

					if (model_type == 2 || model_type == 4) {
						l1_pinyin = vocab[last_word].pinyin_idx * layer1_size;
					}
					for (c = 0; c < layer1_size; c++)
						neu1e[c] = 0;

					// HIERARCHICAL SOFTMAX
					if (hs) {
						for (d = 0; d < vocab[word].codelen; d++) {
							f = 0;
							l2 = vocab[word].point[d] * layer1_size;

							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++)
								if (model_type == 2 || model_type == 4)
									f += (syn0[c + l1] + pinyin_v[c + l1_pinyin]) * syn1[c + l2];
								else
									f += syn0[c + l1] * syn1[c + l2];

							if (f <= -MAX_EXP)
								continue;
							else if (f >= MAX_EXP)
								continue;
							else
								f = exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

							// 'g' is the gradient multiplied by the learning rate
							g = (1 - vocab[word].code[d] - f) * alpha;

							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++)
								neu1e[c] += g * syn1[c + l2];

							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++)
								if (model_type == 2 || model_type == 4)
									syn1[c + l2] += g * (syn0[c + l1] + pinyin_v[c + l1_pinyin]);
								else
									syn1[c + l2] += g * syn0[c + l1];
						}
					}

					// NEGATIVE SAMPLING
					if (negative > 0) {
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table[(next_random >> 16) % table_size];
								if (target == 0)
									target = next_random % (vocab_size - 1) + 1;
								if (target == word)
									continue;
								label = 0;
							}
							l2 = target * layer1_size;
							f = 0;

							for (c = 0; c < layer1_size; c++)
								if (model_type == 2 || model_type == 4)
									f += (syn0[c + l1] + pinyin_v[c + l1_pinyin]) * syn1neg[c + l2];
								else
									f += syn0[c + l1] * syn1neg[c + l2];

							if (f > MAX_EXP)
								g = (label - 1) * alpha;
							else if
								(f < -MAX_EXP) g = (label - 0) * alpha;
							else
								g = (label - exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++)
								neu1e[c] += g * syn1neg[c + l2];

							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++)
								if (model_type == 2 || model_type == 4)
									syn1neg[c + l2] += g * (syn0[c + l1] + pinyin_v[c + l1_pinyin]);
								else
									syn1neg[c + l2] += g * syn0[c + l1];
						}
					}

					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++)
						syn0[c + l1] += neu1e[c];
					if (model_type == 2 || model_type == 4)
						pinyin_v[c + l1_pinyin] += neu1e[c];
				}
		}
		sentence_pos++;
		if (sentence_pos >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void trainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	if (pt == NULL) {
		fprintf(stderr, "cannot allocate memory for threads\n");
		exit(1);
	}
	printf("Starting training using file %s\n", train_word_file);
	starting_alpha = alpha;
	if (read_vocab_file[0] != 0)
		readVocab();
	else
		learnVocabFromTrainFile();

	if (save_vocab_file[0] != 0)
		saveVocab(WORD_TYPE);

	if (model_type == 2 || model_type == 4)
		if (save_pinyin_vocab_file[0] != 0)
			saveVocab(PINYIN_TYPE);

	if (output_file[0] == 0)
		return;

	initNet();
	if (negative > 0)
		initUnigramTable();

	start = clock();
	for (a = 0; a < num_threads; a++)
		pthread_create(&pt[a], NULL, trainModelThread, (void *)a);

	for (a = 0; a < num_threads; a++)
		pthread_join(pt[a], NULL);

	fo = fopen(output_file, "wb");

	if (fo == NULL) {
		fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
		exit(1);
	}
	if (classes == 0) {
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			if (vocab[a].word != NULL) {
				fprintf(fo, "%s ", vocab[a].word);
			}
			if (binary)
				for (b = 0; b < layer1_size; b++)
					if (model_type == 2 || model_type == 4){
						real temp_v = syn0[a * layer1_size + b] + pinyin_v[vocab[a].pinyin_idx * layer1_size + b];
						fwrite(&temp_v, sizeof(real), 1, fo);
					}
					else
						fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else
				for (b = 0; b < layer1_size; b++)
					if (model_type == 2 || model_type == 4)
						fprintf(fo, "%lf ", (syn0[a * layer1_size + b] + pinyin_v[vocab[a].pinyin_idx * layer1_size + b]));
					else
						fprintf(fo, "%lf ", syn0[a * layer1_size + b]);

			fprintf(fo, "\n");
		}
	} else {
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		if (centcn == NULL) {
			fprintf(stderr, "cannot allocate memory for centcn\n");
			exit(1);
		}
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++)
				cent[b] = 0;
			for (b = 0; b < clcn; b++)
				centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++) {
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
					centcn[cl[c]]++;
				}
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++)
			fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
	free(table);
	free(pt);
	destroyVocab();
}

int argPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
		printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
		return 0;
	}
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	save_pinyin_vocab_file[0] = 0;
	read_pinyin_vocab_file[0] = 0;
	if ((i = argPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-train-word", argc, argv)) > 0) strcpy(train_word_file, argv[i + 1]);
	if ((i = argPos((char *)"-train-pinyin", argc, argv)) > 0) strcpy(train_pinyin_file, argv[i + 1]);
	if ((i = argPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = argPos((char *)"-save-pinyin-vocab", argc, argv)) > 0) strcpy(save_pinyin_vocab_file, argv[i + 1]);
	if ((i = argPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = argPos((char *)"-read-pinyin-vocab", argc, argv)) > 0) strcpy(read_pinyin_vocab_file, argv[i + 1]);
	if ((i = argPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = argPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = argPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = argPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-model-type", argc, argv)) > 0) model_type = atoi(argv[i + 1]);

	printf("model_type: %d\n", model_type);
	/* printf("train_word_file: %s\n", train_word_file); */
	/* printf("output: %s\n", output_file); */

	vocab = (struct VocabWord *)calloc(vocab_max_size, sizeof(struct VocabWord));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

	if (model_type == 2 || model_type == 4){
		pinyin_vocab = (struct VocabPinyin*)calloc(pinyin_vocab_max_size, sizeof(struct VocabPinyin));
		pinyin_vocab_hash = (int *)calloc(pinyin_vocab_hash_size, sizeof(int));
	}

	exp_table = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	if (exp_table == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(1);
	}

	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		// sigmoid function: f(x) = 1 / (1 + e^-x)  or f(x) = e^x / (1 + e^x)
		// sigmoid derivative: f(x)' = f(x)(1 - f(x))
		exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		exp_table[i] = exp_table[i] / (exp_table[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	trainModel();
	destroyNet();
	if (vocab_hash != NULL)
		free(vocab_hash);
	if (pinyin_vocab_hash != NULL)
		free(pinyin_vocab_hash);
	free(exp_table);
	return 0;
}
