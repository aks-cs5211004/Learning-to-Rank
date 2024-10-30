#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: glove-gen-rerank.sh [query-file] [top-100-file] [collection-file] [glove-embeddings-file] [output-file] [expansions-file]"
    exit 1
fi

# Assign input arguments to variables
query_file=$1
top_100_file=$2
collection_file=$3
glove_embeddings_file=$4
output_file=$5
expansions_file=$6

# Run the Python script for the GloVe generic model
python3 glove_gen.py --query_file "$query_file" --top_100_file "$top_100_file" --collection_file "$collection_file" --glove_embeddings_file "$glove_embeddings_file" --output_file "$output_file" --expansions_file "$expansions_file"
