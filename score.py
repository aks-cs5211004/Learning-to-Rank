import argparse
from collections import defaultdict
from math import log2


# Function to parse results file (results after ranking)
def parse_results_file(results_file_path):
    results = defaultdict(dict)  # query_number -> rank -> cord_id
    with open(results_file_path, 'r') as results_file:
        for line in results_file.readlines():
            if line.strip():
                query_number, ignore, cord_id, rank, score, run_id = line.split()
                query_number = int(query_number)
                rank = int(rank)
                results[query_number][rank] = cord_id
    return results

# Function to parse top100docs.tsv (results before ranking)
def parse_top100_file(top100_file_path):
    top100_results = defaultdict(list)  # query_number -> list of cord_ids
    with open(top100_file_path, 'r') as top100_file:
        for line in top100_file.readlines():
            if line.startswith('[query_id]'):
                continue
            if line.strip():
                query_number, cord_id, score = line.split()
                query_number = int(query_number)
                top100_results[query_number].append(cord_id)
    return top100_results

# Function to calculate discounted cumulative gain
def dcg(scores, rank):
    return sum(score / log2(idx + 2) for idx, score in enumerate(scores[:rank]))

# Function to calculate nDCG
def ndcg_at_k(output_scores, ideal_scores, k):
    output_dcg = dcg(output_scores, k)
    ideal_dcg = dcg(ideal_scores, k)
    return output_dcg / ideal_dcg if ideal_dcg > 0 else 0

# Argument parser setup
parser = argparse.ArgumentParser(description="Compute NDCG scores for different Queries")
parser.add_argument('--gold_query_relevances_path', required=True, type=str, help="Path to the gold query relevances file in TREC format")
parser.add_argument('--results_path', required=True, type=str, help="Path to query results files in TREC format")
parser.add_argument('--top100_path', required=True, type=str, help="Path to the top100 documents file before ranking")

args = parser.parse_args()

gold_query_relevances_path = args.gold_query_relevances_path
results_path = args.results_path
top100_path = args.top100_path

# Parse gold relevances
gold_results = defaultdict(dict)  # query_number -> cord_id -> score
with open(gold_query_relevances_path, 'r') as gold_file:
    for line in gold_file.readlines():
        if line.startswith('[query_id]'):
            continue
        if line.strip():
            query_number, cord_id, relevance, run = line.split()
            query_number = int(query_number)
            relevance = int(relevance)
            gold_results[query_number][cord_id] = relevance

# Parse results after ranking
output_results = parse_results_file(results_file_path=results_path)

# Parse top100 documents before ranking
top100_results = parse_top100_file(top100_file_path=top100_path)

query_numbers = list(gold_results.keys())

# Initialize score accumulators
ndcg5_sum, ndcg10_sum, ndcg50_sum = 0, 0, 0
ndcg5_top100_sum, ndcg10_top100_sum, ndcg50_top100_sum = 0, 0, 0

def get_ideal_scores(query_number):
    return sorted([relevance for relevance in gold_results[query_number].values()], reverse=True)

# Process each query and calculate scores
for query_number in query_numbers:
    if query_number not in output_results or query_number not in top100_results:
        print(f"Warning: No results or top100 docs for query {query_number}. Skipping...")
        continue

    # Ideal relevance scores (sorted gold scores)
    ideal_scores = get_ideal_scores(query_number)

    # Calculate output scores (after ranking)
    output_scores = [gold_results[query_number].get(output_results[query_number][rank], 0) for rank in sorted(output_results[query_number].keys())]

    # Calculate top100 scores (before ranking)
    top100_scores = [gold_results[query_number].get(cord_id, 0) for cord_id in top100_results[query_number]]

    # Compute NDCG for both top100 (before ranking) and output results (after ranking)
    ndcg5 = ndcg_at_k(output_scores, ideal_scores, 5)
    ndcg10 = ndcg_at_k(output_scores, ideal_scores, 10)
    ndcg50 = ndcg_at_k(output_scores, ideal_scores, 50)

    ndcg5_top100 = ndcg_at_k(top100_scores, ideal_scores, 5)
    ndcg10_top100 = ndcg_at_k(top100_scores, ideal_scores, 10)
    ndcg50_top100 = ndcg_at_k(top100_scores, ideal_scores, 50)

    # Accumulate results for averaging
    ndcg5_sum += ndcg5
    ndcg10_sum += ndcg10
    ndcg50_sum += ndcg50

    ndcg5_top100_sum += ndcg5_top100
    ndcg10_top100_sum += ndcg10_top100
    ndcg50_top100_sum += ndcg50_top100

    # Print individual query results
    print(f"Query: {query_number}")
    print(f"\tBefore Ranking - NDCG@5: {ndcg5_top100:.4f}, NDCG@10: {ndcg10_top100:.4f}, NDCG@50: {ndcg50_top100:.4f}")
    print(f"\tAfter Ranking  - NDCG@5: {ndcg5:.4f}, NDCG@10: {ndcg10:.4f}, NDCG@50: {ndcg50:.4f}")

# Average NDCG Scores
num_queries = len(query_numbers)

avg_ndcg5 = ndcg5_sum / num_queries
avg_ndcg10 = ndcg10_sum / num_queries
avg_ndcg50 = ndcg50_sum / num_queries

avg_ndcg5_top100 = ndcg5_top100_sum / num_queries
avg_ndcg10_top100 = ndcg10_top100_sum / num_queries
avg_ndcg50_top100 = ndcg50_top100_sum / num_queries

# Print average scores
print(f"\nAverage NDCG Scores Before Ranking:")
print(f"\tNDCG@5: {avg_ndcg5_top100:.4f}, NDCG@10: {avg_ndcg10_top100:.4f}, NDCG@50: {avg_ndcg50_top100:.4f}")

print(f"\nAverage NDCG Scores After Ranking:")
print(f"\tNDCG@5: {avg_ndcg5:.4f}, NDCG@10: {avg_ndcg10:.4f}, NDCG@50: {avg_ndcg50:.4f}")
