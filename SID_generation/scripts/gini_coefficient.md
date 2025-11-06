# Gini Coefficient Calculation for Semantic IDs

This project aims to calculate the Gini Coefficient for semantic IDs using SQL queries and a Python script.

## 1. Data Preparation

First, we need a base data table containing `item_id` and `sid` (semantic ID). Let's assume this table is named `cobebook_result`.

### 1.1 SQL Queries

The following SQL queries are used to extract the necessary information from the raw data for Gini coefficient calculation:

#### 1.1.1 Semantic ID Result Table

```sql
CREATE TABLE cobebook_result
(
    item_id  BIGINT      -- Item ID
    ,sid     STRING      -- Semantic ID
)
;
```

#### 1.1.2 Count for Each Semantic ID

This query counts the occurrences of each `sid` (semantic ID).

```sql
CREATE TABLE leaf_sid_cnt
SELECT  sid           -- Semantic ID
        ,COUNT(*) AS cnt -- Count of occurrences for this Semantic ID
FROM    cobebook_result
GROUP BY sid
;
```

#### 1.1.3 Frequency of Each Count

This query counts how many `sid`s have the same `cnt` value. For example, if 10 different `sid`s each appeared 5 times, then `cnt_freq` for `cnt=5` would be 10.

```sql
CREATE TABLE leaf_sid_cnt_freq
SELECT  cnt           -- Number of occurrences for a semantic ID
        ,cnt_freq      -- Number of semantic IDs that have this occurrence count
FROM    (
            SELECT  cnt
                    ,COUNT(1) AS cnt_freq -- Calculate how many rows have each cnt value
            FROM    leaf_sid_cnt
            GROUP BY cnt
        ) AS subquery
ORDER BY cnt            -- Order by count in ascending order
;
```

**Important Note:** Export the data from the `leaf_sid_cnt_freq` table to a local file, for example, named `row_summary.txt`. The file format should be `cnt$$||$$cnt_freq`, e.g., `1$$||$$100`.

## 2. Gini Coefficient Calculation

After downloading the data from the `leaf_sid_cnt_freq` table, you can use the following Python script to calculate the Gini coefficient.

### 2.1 Python Script (`calculate_gini.py`)

```python
import numpy as np

# Load input data
# List of [count, frequency of count], already sorted in ascending order
# Assumes the data file is named 'row_summary.txt', with each line in 'count$$||$$frequency' format
with open('row_summary.txt', 'r') as f:
    summary = []
    for line in f.readlines():
        # Strip newline characters and split by delimiter
        parts = line.strip().split('$$||$$')
        # Ensure there are exactly two parts and convert to integers
        if len(parts) == 2:
            summary.append(list(map(int, parts)))
        else:
            print(f"Warning: Skipping malformed line: {line.strip()}")

# Convert the list to a NumPy array for subsequent calculations
summary = np.array(summary)

def simplified_dot_product(summary_array):
    """
    Calculates the simplified dot product used in Gini coefficient calculation.
    summary_array: NumPy array, each row is [cnt, cnt_freq].
                   cnt: The number of occurrences for a specific sid.
                   cnt_freq: The number of sids that have that specific count.
    """
    # Extract cnt and cnt_freq columns
    counts_of_sid = summary_array[:, 0]  # j: number of occurrences for a semantic ID
    freq_of_counts = summary_array[:, 1] # n_j: number of semantic IDs having that count

    N = np.sum(freq_of_counts) # Total number of unique SIDs (not total items)

    total_dot_product = 0
    cumulative_freq_before = 0 # Cumulative number of sids (before the current cnt_freq)

    # Iterate through each (j, n_j) pair
    for j, n_j in summary_array:
        if n_j == 0:
            continue

        # Calculate the boundaries for the arithmetic series sum
        # (N - cumulative_freq_before) is the rank of the first element in the current group
        # (N - cumulative_freq_before - n_j + 1) is the rank of the last element in the current group
        first_term = N - cumulative_freq_before
        last_term = N - cumulative_freq_before - n_j + 1

        # Sum of an arithmetic series formula: (first_term + last_term) * number_of_terms / 2
        sum_of_b_block = (first_term + last_term) * n_j // 2

        # Multiply by the current j (cnt value)
        block_dot_product = j * sum_of_b_block
        total_dot_product += block_dot_product

        # Update the cumulative count of sids
        cumulative_freq_before += n_j

    return total_dot_product, N

# Ensure the summary array is not empty
if summary.size == 0:
    print("Error: Input summary data is empty. Cannot calculate Gini coefficient.")
else:
    # Call the function to calculate the simplified dot product and N
    result_simplified_dot_product, N_sids = simplified_dot_product(summary)

    print(f"The simplified dot product (2 * sum(j * rank_j)) is: {result_simplified_dot_product}")
    print(f"Total number of unique SIDs (N) is: {N_sids}")

    # Calculate the total number of items (sum(j * n_j))
    # This step calculates the total sum of all item_ids, i.e., sum(cnt * cnt_freq)
    # For example: (1*100) + (2*50) + (3*20) ...
    sum_j_times_nj = np.dot(summary[:, 0], summary[:, 1])

    # Ensure the denominator is not zero
    if N_sids == 0 or sum_j_times_nj == 0:
        print("Error: Denominator for Gini calculation is zero. Cannot calculate Gini coefficient.")
    else:
        # Gini Coefficient formula
        gini = (N_sids + 1) / N_sids - (2 * result_simplified_dot_product / (N_sids * sum_j_times_nj))
        print(f"The Gini Coefficient is: {gini}")

```

### 2.2 Execution Steps

1.  **Execute SQL:** Follow the order in Section 1.1 to execute the SQL queries and generate the `leaf_sid_cnt_freq` table.
2.  **Export Data:** Export the data from the `leaf_sid_cnt_freq` table to a file named `row_summary.txt`. Ensure the file content format is `cnt$$||$$cnt_freq`, for example:
    ```
    1$$||$$100
    2$$||$$50
    3$$||$$20
    ...
    ```
3.  **Run Python Script:** Save the above Python code as `calculate_gini.py`, then run it in your terminal:
    ```bash
    python calculate_gini.py
    ```
4.  **View Results:** The script will output the calculated simplified dot product and the Gini Coefficient.
