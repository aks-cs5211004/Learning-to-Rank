import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

# Define varying values for W2V_LAMBDA
w2v_lambda_values = [0.2]  # Adjust W2V_LAMBDA values here

# Initialize dictionaries to hold scores for each W2V_LAMBDA setting
before_ndcg5_scores = {0.2: None, 0.5: None, 0.7: None}
before_ndcg10_scores = {0.2: None, 0.5: None, 0.7: None}
before_ndcg50_scores = {0.2: None, 0.5: None, 0.7: None}
after_ndcg5_scores = {0.2: None, 0.5: None, 0.7: None}
after_ndcg10_scores = {0.2: None, 0.5: None, 0.7: None}
after_ndcg50_scores = {0.2: None, 0.5: None, 0.7: None}

# File paths
gold_query_relevances_path = 'qrels.tsv'  # Adjust this path
results_path = 'glove_gen.out'  # This will be created after reranking
top100_path = 'top100docs.tsv'  # Adjust this path

for w2v_lambda in w2v_lambda_values:
    # Write the new value to constants.py
    with open('constants_gen.py', 'r') as file:
        lines = file.readlines()
    
    with open('constants_gen.py', 'w') as file:
        for line in lines:
            if line.startswith('W2V_LAMBDA'):
                print(f"Setting W2V_LAMBDA to {w2v_lambda}")
                file.write(f'W2V_LAMBDA = {w2v_lambda}\n')
            else:
                file.write(line)
                
    # Run the reranking command
    subprocess.run(["bash", "glove-gen_rerank.sh", "queries.tsv", "top100docs.tsv", "docs.tsv", "glove.6B.300d.txt", "glove_gen.out", "glove_gen.expansions"])
    
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

    # Store the extracted values for the current W2V_LAMBDA setting
    if (before_ndcg5 is not None and after_ndcg5 is not None):
        print(f"Extracted: Before NDCG@5 = {before_ndcg5}, After NDCG@5 = {after_ndcg5}")
        before_ndcg5_scores[w2v_lambda] = before_ndcg5
        before_ndcg10_scores[w2v_lambda] = before_ndcg10
        before_ndcg50_scores[w2v_lambda] = before_ndcg50
        after_ndcg5_scores[w2v_lambda] = after_ndcg5
        after_ndcg10_scores[w2v_lambda] = after_ndcg10
        after_ndcg50_scores[w2v_lambda] = after_ndcg50
    else:
        print(f"Warning: Could not extract nDCG scores for W2V_LAMBDA={w2v_lambda}")

# Plotting the results
plt.figure(figsize=(10, 6))
print("NDCG@5 scores before reranking:", before_ndcg5_scores)
print("NDCG@10 scores before reranking:", before_ndcg10_scores)
print("NDCG@50 scores before reranking:", before_ndcg50_scores)
print("NDCG@5 scores after reranking:", after_ndcg5_scores)
print("NDCG@10 scores after reranking:", after_ndcg10_scores)
print("NDCG@50 scores after reranking:", after_ndcg50_scores)

# Plotting NDCG@5 scores
plt.plot(w2v_lambda_values, [before_ndcg5_scores[0.2], before_ndcg5_scores[0.5], before_ndcg5_scores[0.7]], label='Before Reranking NDCG@5', marker='o', color='blue')
plt.plot(w2v_lambda_values, [after_ndcg5_scores[0.2], after_ndcg5_scores[0.5], after_ndcg5_scores[0.7]], label='After Reranking NDCG@5', marker='o', color='lightblue')

# Plotting NDCG@10 scores
plt.plot(w2v_lambda_values, [before_ndcg10_scores[0.2], before_ndcg10_scores[0.5], before_ndcg10_scores[0.7]], label='Before Reranking NDCG@10', marker='o', color='green')
plt.plot(w2v_lambda_values, [after_ndcg10_scores[0.2], after_ndcg10_scores[0.5], after_ndcg10_scores[0.7]], label='After Reranking NDCG@10', marker='o', color='lightgreen')

# Plotting NDCG@50 scores
plt.plot(w2v_lambda_values, [before_ndcg50_scores[0.2], before_ndcg50_scores[0.5], before_ndcg50_scores[0.7]], label='Before Reranking NDCG@50', marker='o', color='orange')
plt.plot(w2v_lambda_values, [after_ndcg50_scores[0.2], after_ndcg50_scores[0.5], after_ndcg50_scores[0.7]], label='After Reranking NDCG@50', marker='o', color='red')

# Customizing the plot
plt.xlabel('W2V_LAMBDA Value')
plt.ylabel('NDCG Scores')
plt.title('NDCG Scores Before and After Reranking (Varying W2V_LAMBDA)')
plt.xticks(w2v_lambda_values, w2v_lambda_values)  # Map W2V_LAMBDA values to ticks
plt.legend()
plt.grid()
plt.savefig('ndcg_scores_W2V_LAMBDA_plot.png')
plt.show()
