/*
 * pinyin.c
 * Copyright (C) 2019 LeonTao 
 *
 * Distributed under terms of the MIT license.
 */

/* #include "pinyin.h" */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Build a Tire for Pinyin search.
 **/

#define MAX_STRING 100
#define CHINESE_CHRACTER_COUNT 3500

struct TrieNode {
    char *word;
    TrieNode *child[CHINESE_CHRACTER_COUNT];
    bool isEnd;
};

char pinyin_file[MAX_STRING];

// 1, read pinyin file
void ReadWord(char *word, FILE *fin) {
    // Reads a single word from a file, assuming space + tab + EOL to be word boundaries
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) 
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') 
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else 
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) 
            a--;   // Truncate too long words
    }
    word[a] = 0;
}


// 2, build trie
void buildPinyinTrie(pinyin_file){
    FILE *fin = fopen(pinyin_file, "r");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

}


// 3, search by word
void searchPinyin(TrieNode *root, char *word){
    tmpRoot = root;
    len = strlen(word);
    int i;
    for (i = 0; i < len; i++){
        int ord = 
        if (root)
        
    }
}












