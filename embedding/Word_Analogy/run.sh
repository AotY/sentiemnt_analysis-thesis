#!/usr/bin/env bash


FIELD=news

# declare -a model_types=(1 2 3 4)
declare -a model_types=(1 2 3)
declare -a sizes=(100 200)
declare -a csms=("cbow" "sg") # continuous skip-gram models, (cbow, sg)

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

			vector_path=word2vec/data/merge.$FIELD.$mt.$csm.$cea.$size.bin
            echo $vector_path
            python Word_Analogy/word_analogy.py --vector $vector_path --analogy ./Word_Analogy/Data/analogy.txt --binary

			sleep 1

		done
	done
done
