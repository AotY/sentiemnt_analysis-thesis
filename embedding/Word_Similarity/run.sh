#!/usr/bin/env bash

FIELD=web

declare -a model_types=(1 2 3 4)
# declare -a sizes=(100 200)
# declare -a model_types=(4)
declare -a sizes=(100)
declare -a csms=("cbow" "sg") # continuous skip-gram models, (cbow, sg)
# declare -a hses=(0 1) # continuous skip-gram models, (cbow, sg)
declare -a hses=(0) # hierarchical softmax, or negtaive

mt=1
cbow=0
size=100
cea=negative
hs=0
f=1

for mt in "${model_types[@]}"
do
	for size in "${sizes[@]}"
	do
		for csm in "${csms[@]}"
		do
			if [ "$csm" == "sg" ]; then
				cbow=0
                if [ "$mt" -gt 2 ]; then
                    continue
                fi
			else
				cbow=1
			fi

			if [ "$FIELD" == "web" ]; then
				f=1
			else
				f=2
			fi

            for hs in "${hses[@]}"
            do
                if [ "$hs" -eq 0 ]; then
                    cea=negative
                else
                    cea=hs
                fi

                vector_path=./word2vec/data/merge.$FIELD.$mt.$csm.$cea.$size.bin
                echo $vector_path
                python Word_Similarity/word_similarity.py --vector $vector_path --similarity ./Word_Similarity/Data/wordsim-240.txt --binary

                sleep 1

            done
		done
	done
done
