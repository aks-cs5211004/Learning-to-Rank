## Directory Information:

- *constants.py*: contains hyperparameters used for Task 0 and 1
- *constants_gen.py*: contains hyperparameters used for Task 2.1 and 2.2
- *lm.py*: Main Python script used by Task-0
- *lm_gen.py*: Copy of lm.py which reads parameters from *constants_gen.py*
- *w2v_local.py*: Main Python script used by Task-1
- *w2v_gen.py*: Main Python script used by Task-2.1
- *glove_gen.py*: Main Python script used by Task-2.2
- *build.sh*: initial building
- *w2v-local_rerank.sh*
- *w2v-gen_rerank.sh*
- *glove-gen_rerank.sh*
- *plot_local.py*: For plotting graphs for Task 1
- *plot_w2v.py*: For plotting graphs for Task 2.1
- *plot_glove.py*: For plotting graphs for Task 2.2
- *score.py*: calculating scores
- README.md
- Report.pdf

## How to go about Running the Project?

### Running build.sh
```
bash build.sh
```


### Running Task-0 (Pseudo Relevance Language Modelling)
```
python3 lm.py --query_file <file> --top_100_file <file> --collection_file <file> --output_file <file>
```
Eg.
```
python3 lm.py --query_file queries.tsv --top_100_file top100docs.tsv --collection_file docs.tsv --output_file lm.out
```

### Running Task-1 (Local Word2Vec Embeddings)
```
bash w2v-local_rerank.sh [query-file] [top-100-file] [collection-dir] [output-file] [expansions-file]
```
Eg.
```
bash w2v-local_rerank.sh queries.tsv top100docs.tsv docs.tsv w2v_local.out w2v_local.expansions
```

### Running Task-2.1 (Generic Word2Vec Embeddings)
```
bash w2v-gen_rerank.sh [query-file] [top-100-file] [collection-dir] [word-embeddings-file] [output-file] [expansions-file]
```
Eg.
```
bash w2v-gen_rerank.sh queries.tsv top100docs.tsv docs.tsv word2vec.300d.txt w2v_gen.out w2v_gen.expansions
```

### Running Task-2.2 (Generic Glove Embeddings)
```
bash glove-gen_rerank.sh [query-file] [top-100-file] [collection-dir] [glove-embeddings-file] [output-file] [expansions-file]
```
Eg.
```
bash glove-gen_rerank.sh queries.tsv top100docs.tsv docs.tsv glove.6B.300d.txt glove_gen.out glove_gen.expansions
```



## The Evaluations!

### Running TREC Eval
```
./trec_eval-9.0.7/trec_eval -m ndcg -m ndcg_cut.5,10,50 <Query-Results-File> <Output-To-Be-Tested>
```
eg. 
```
./trec_eval-9.0.7/trec_eval -m ndcg -m ndcg_cut.5,10,50 qrels.tsv glove_gen.out
```

### Running Custom Eval (*score.py*)
```
python3 score.py --gold_query_relevances_path <path> --results_path <path> --top100_path <path>
```

eg.
```
python3 score.py --gold_query_relevances_path qrels.tsv --results_path glove_gen.out --top100_path top100docs.tsv
```

-----



## Running Individual Python Files

Task 0

```
python3 lm.py --query_file=queries.tsv --top_100_file=top100docs.tsv --collection_file=docs.tsv --output_file=w2v_local.out
```

Task 1

```
python3 w2v_local.py --query_file=queries.tsv --top_100_file=top100docs.tsv --collection_file=docs.tsv --output_file=w2v_local.out --expansions_file=w2v_local.expansions
```

Task 2.1

```
python3 w2v_gen.py --query_file=queries.tsv --top_100_file=top100docs.tsv --collection_file=docs.tsv --w2v_embeddings_file=word2vec.300d.txt --output_file=w2v_gens.out --expansions_file=w2v_gens.expansions
```

Task 2.2

```
python3 glove_gen.py --query_file=queries.tsv --top_100_file=top100docs.tsv --collection_file=docs.tsv --glove_embeddings_file=glove.6B.300d.txt --output_file=w2v_gens.out --expansions_file=w2v_gens.expansions
```

