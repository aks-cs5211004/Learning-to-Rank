import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

# Define varying values for DIRICHLET_MU
mu_values = [200]  # Example range of DIRICHLET_MU values

# Initialize dictionaries to hold scores for each DIRICHLET_MU setting
before_ndcg5_scores = {}
before_ndcg10_scores = {}
before_ndcg50_scores = {}
after_ndcg5_scores = {}
after_ndcg10_scores = {}
after_ndcg50_scores = {}

# File paths
gold_query_relevances_path = 'qrels.tsv'  # Adjust this path
results_path = 'w2v_local.out'  # This will be created after reranking
top100_path = 'top100docs.tsv'  # Adjust this path

for mu_value in mu_values:
    # Write the new value to constants.py
    with open('constants.py', 'r') as file:
        print(f"Setting DIRICHLET_MU to {mu_value}")
        lines = file.readlines()
    
    with open('constants.py', 'w') as file:
        for line in lines:
            if line.startswith('DIRICHILET_MU'):
                print(f"Setting DIRICHILET_MU to {mu_value}")
                file.write(f'DIRICHILET_MU = {mu_value}\n')
            else:
                file.write(line)

    # Run the reranking command
    subprocess.run(["bash", "w2v-local_rerank.sh", "queries.tsv", "top100docs.tsv", "docs.tsv", "w2v_local.out", "w2v_local.expansions"])
    
    # Calculate nDCG before and after reranking
    result = subprocess.run(["python3", "score.py", "--gold_query_relevances_path", gold_query_relevances_path, 
                             "--results_path", results_path, "--top100_path", top100_path],
                            capture_output=True, text=True)
    
    # After running score.py, capture the output
    output = result.stdout

    # Extract NDCG scores from the output
    before_ndcg5 = before_ndcg10 = before_ndcg50 = None
    after_ndcg5 = after_ndcg10 = after_ndcg50 = None
    
    output_lines = output.splitlines()
    
    for i, line in enumerate(output_lines):
        if "Average NDCG Scores Before Ranking:" in line:
            if i + 1 < len(output_lines):  # Check if the next line exists
                before_line = output_lines[i + 1]
                before_ndcg5 = float(before_line.split("NDCG@5: ")[-1].split(",")[0].strip())
                before_ndcg10 = float(before_line.split("NDCG@10: ")[-1].split(",")[0].strip())
                before_ndcg50 = float(before_line.split("NDCG@50: ")[-1].split(",")[0].strip())
                
        elif "Average NDCG Scores After Ranking:" in line:
            if i + 1 < len(output_lines):  # Check if the next line exists
                after_line = output_lines[i + 1]
                after_ndcg5 = float(after_line.split("NDCG@5: ")[-1].split(",")[0].strip())
                after_ndcg10 = float(after_line.split("NDCG@10: ")[-1].split(",")[0].strip())
                after_ndcg50 = float(after_line.split("NDCG@50: ")[-1].split(",")[0].strip())

    # Store the extracted values for the current DIRICHLET_MU setting
    if (before_ndcg5 is not None and after_ndcg5 is not None):
        print(f"Extracted: Before NDCG@5 = {before_ndcg5}, After NDCG@5 = {after_ndcg5}")
        before_ndcg5_scores[mu_value] = before_ndcg5
        before_ndcg10_scores[mu_value] = before_ndcg10
        before_ndcg50_scores[mu_value] = before_ndcg50
        after_ndcg5_scores[mu_value] = after_ndcg5
        after_ndcg10_scores[mu_value] = after_ndcg10
        after_ndcg50_scores[mu_value] = after_ndcg50
    else:
        print(f"Warning: Could not extract nDCG scores for DIRICHLET_MU={mu_value}")

print("NDCG@5 scores before reranking:", before_ndcg5_scores)
print("NDCG@10 scores before reranking:", before_ndcg10_scores)
print("NDCG@50 scores before reranking:", before_ndcg50_scores)
print("NDCG@5 scores after reranking:", after_ndcg5_scores)
print("NDCG@10 scores after reranking:", after_ndcg10_scores)
print("NDCG@50 scores after reranking:", after_ndcg50_scores)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plotting NDCG@5 scores
plt.plot(mu_values, [before_ndcg5_scores[mu] for mu in mu_values], label='Before Reranking NDCG@5', marker='o', color='blue')
plt.plot(mu_values, [after_ndcg5_scores[mu] for mu in mu_values], label='After Reranking NDCG@5', marker='o', color='lightblue')

# Plotting NDCG@10 scores
plt.plot(mu_values, [before_ndcg10_scores[mu] for mu in mu_values], label='Before Reranking NDCG@10', marker='o', color='green')
plt.plot(mu_values, [after_ndcg10_scores[mu] for mu in mu_values], label='After Reranking NDCG@10', marker='o', color='lightgreen')

# Plotting NDCG@50 scores
plt.plot(mu_values, [before_ndcg50_scores[mu] for mu in mu_values], label='Before Reranking NDCG@50', marker='o', color='orange')
plt.plot(mu_values, [after_ndcg50_scores[mu] for mu in mu_values], label='After Reranking NDCG@50', marker='o', color='red')

# Customizing the plot
plt.xlabel('DIRICHLET_MU Value')
plt.ylabel('NDCG Scores')
plt.title('NDCG Scores Before and After Reranking (Varying DIRICHLET_MU)')
plt.xticks(mu_values, [str(mu) for mu in mu_values])
plt.legend()
plt.grid()
plt.savefig('ndcg_scores_dirichlet_mu_plot.png')
plt.show()
