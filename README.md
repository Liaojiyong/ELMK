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

Train our multi-path encoder on subgraphs. Before training, load the [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model from Hugging Face. Note that you should download the models (including the file tokenizer.model and the folder that store the parameters of the all-mpnet-base-v2) in folder .\sentence-transformers. For example, you may have the following file structure.

|-- sentence-transformers
    |-- all-mpnet-base-v2
        |-- config.json
        |-- modules.json
        |-- pytorch_model.json
        |-- sentence_bert_config.json
        |-- tokenizer_config.json
        |-- tokenizer.json
        |-- 1_Pooling
            |-- config.json

    
Run

    cd MetaQA/encoder
    python pretrain_encoder.py --hop <hop>
    cd ..

## 5. Evaluation

To test the trained model:

Run

    cd MetaQA/test
    python evaluate.py --hop <hop> --k <Tok-k>
    cd ..

## 6. Experimental Settings

We have incorporated two baseline methods and benchmark datasets:

| Baseline | Paper                                                                             | Code   |
|----------|-----------------------------------------------------------------------------------|--------|
| KG-GPT   | A General Framework for Reasoning on Knowledge Graphs Using Large Language Models | [KG-GPT](https://github.com/jiho283/KG-GPT) |
| KELP     | Knowledge Graph-Enhanced Large Language Models via Path Selection                 | [KELP](https://github.com/HaochenLiu2000/KELP)|

## Acknowledgment

The dataset, requirements, and data preparation follow the setting of [KG-GPT](https://github.com/jiho283/KG-GPT/). 
Thanks to the authors and developers!
**Thanks for your interest in our work!**
