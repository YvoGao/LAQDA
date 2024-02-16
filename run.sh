cuda=0
commont=Bert-PN-addQ-CE-att
for path in 01 02 03 04 05;
do
    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont


    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont




    python src_org/main.py \
        --dataset 20News \
        --dataFile data/20News/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont


    python src_org/main.py \
        --dataset 20News \
        --dataFile data/20News/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont



    python src_org/main.py \
        --dataset Amazon \
        --dataFile data/Amazon/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont


    python src_org/main.py \
        --dataset Amazon \
        --dataFile data/Amazon/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont



    python src_org/main.py \
        --dataset Reuters \
        --dataFile data/Reuters/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 5 \
        --k 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numQShot 15 \
        --numFreeze 6 \
        --commont=$commont


    python src_org/main.py \
        --dataset Reuters \
        --dataFile data/Reuters/few_shot/${path} \
        --fileVocab=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelConfig=/mnt/Yvo_data/yvo/model/BERT/config.json \
        --fileModel=/mnt/Yvo_data/yvo/model/BERT \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numQShot 15 \
        --numFreeze 6 \
        --commont=$commont
done
