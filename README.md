## 1. Datasets

The dataset, requirements, and data preparation follow the setting of [KG-GPT](https://github.com/jiho283/KG-GPT/). 

Download [FactKG](https://github.com/jiho283/FactKG) and [MetaQA](https://github.com/yuyuz/MetaQA) here.

Place the files or folders `kb.txt`, `1-hop/vanilla`, `2-hop/vanilla`  `3-hop/vanilla` under `./MetaQA/data`.

For data preprocessing.

Run

    cd MetaQA/data
    python preprocess.py --hop <hop>
    cd ..

## 2. Openai Key

Write your own OpenAI API key in factkg/openai_api_key.txt and MetaQA/openai_api_key.txt and save them.

## 3. Building of the Training Sub-dataset

To build the specific training sub-dataset from the original datasets:

Run

    cd MetaQA/encoder
    python subgraph_builder.py --setting train --hop <hop>
    python subgraph_builder.py --setting dev --hop <hop>
    cd ..


## 4. Training

Train our multi-path encoder on subgraphs. Before training, load the all-mpnet-base-v2 model (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)  from Hugging Face.
[all-mpnet-base-v2 model](https://huggingface.co/sentence-transformers/all-mpnet-base-v)

Run

    cd MetaQA/encoder
    python pretrain_encoder.py --hop <hop>
    cd ..

## 4. Evaluation

To test the trained model:

Run

    cd MetaQA/test
    python evaluate.py --hop <hop> --k <Tok-k>
    cd ..

## 5. Acknowledgment

The dataset, requirements, and data preparation follow the setting of [KG-GPT](https://github.com/jiho283/KG-GPT/). 
Thanks to the authors and developers!
**Thanks for your interest in our work!**
