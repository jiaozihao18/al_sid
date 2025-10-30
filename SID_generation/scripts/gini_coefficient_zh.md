# 计算语义ID的基尼系数 (Gini Coefficient for Semantic IDs)

本项目旨在通过 SQL 查询和 Python 脚本计算语义ID（semantic ID）的基尼系数（Gini Coefficient）。

## 1. 数据准备 (Data Preparation)

首先，我们需要一张包含 `item_id` 和 `sid` (semantic ID) 的基础数据表。假设该表名为 `cobebook_result`。

### 1.1 SQL 查询 (SQL Queries)

以下 SQL 查询用于从原始数据中提取计算基尼系数所需的信息：

#### 1.1.1 语义ID结果表 (Semantic ID Result Table)

```sql
CREATE TABLE cobebook_result
(
    item_id  BIGINT      -- 条目ID
    ,sid     STRING      -- 语义ID
)
;
```

#### 1.1.2 统计每个语义ID的数量 (Count for Each Semantic ID)

该查询将统计每个 `sid` (semantic ID) 出现的次数。

```sql
CREATE TABLE leaf_sid_cnt
SELECT  sid           -- 语义ID
        ,COUNT(*) AS cnt -- 该语义ID出现的次数
FROM    cobebook_result
GROUP BY sid
;
```

#### 1.1.3 统计每个数量的频次 (Frequency of Each Count)

此查询将统计有多少个 `sid` 具有相同的 `cnt` 值。例如，如果有10个不同的 `sid` 各自出现了5次，那么 `cnt=5` 的 `cnt_freq` 就是10。

```sql
CREATE TABLE leaf_sid_cnt_freq
SELECT  cnt           -- 某个语义ID出现的次数
        ,cnt_freq      -- 具有该次数的语义ID的数量
FROM    (
            SELECT  cnt
                    ,COUNT(1) AS cnt_freq -- 计算每个cnt值有多少行
            FROM    leaf_sid_cnt
            GROUP BY cnt
        ) AS subquery
ORDER BY cnt            -- 按次数从小到大排序
;
```

**重要提示:** 将 `leaf_sid_cnt_freq` 表的数据导出到本地文件，例如命名为 `row_summary.txt`。文件格式应为 `cnt$$||$$cnt_freq`，例如 `1$$||$$100`。

## 2. 基尼系数计算 (Gini Coefficient Calculation)

下载 `leaf_sid_cnt_freq` 表的数据后，可以使用以下 Python 脚本计算基尼系数。

### 2.1 Python 脚本 (`calculate_gini.py`) (Python Script)

```python
import numpy as np

# 加载输入数据
# list of [count, frequency of count], 已经排序过，从小到大
# 假设数据文件名为 'row_summary.txt'，每行格式为 'count$$||$$frequency'
with open('row_summary.txt', 'r') as f:
    summary = []
    for line in f.readlines():
        # 清除行末的换行符，并按分隔符拆分
        parts = line.strip().split('$$||$$')
        # 确保只有两个部分，并转换为整数
        if len(parts) == 2:
            summary.append(list(map(int, parts)))
        else:
            print(f"Warning: Skipping malformed line: {line.strip()}")

# 将列表转换为 NumPy 数组以便于后续计算
summary = np.array(summary)

def simplified_dot_product(summary_array):
    """
    计算简化点积，用于基尼系数的计算。
    summary_array: NumPy 数组，每行 [cnt, cnt_freq]。
                   cnt: 某个sid出现的次数。
                   cnt_freq: 具有该次数的sid的数量。
    """
    # 提取cnt和cnt_freq列
    counts_of_sid = summary_array[:, 0]  # j: 语义ID出现的次数
    freq_of_counts = summary_array[:, 1] # n_j: 具有该次数的语义ID的数量

    N = np.sum(freq_of_counts) # 所有的sid总数 (不是item总数)

    total_dot_product = 0
    cumulative_freq_before = 0 # 累积的sid数量（在当前cnt_freq之前）

    # 遍历每个 (j, n_j) 对
    for j, n_j in summary_array:
        if n_j == 0:
            continue

        # 计算等差数列求和的边界
        # (N - cumulative_freq_before) 是当前组的第一个元素的排名
        # (N - cumulative_freq_before - n_j + 1) 是当前组的最后一个元素的排名
        first_term = N - cumulative_freq_before
        last_term = N - cumulative_freq_before - n_j + 1

        # 等差数列求和公式: (首项 + 末项) * 项数 / 2
        sum_of_b_block = (first_term + last_term) * n_j // 2

        # 乘以当前的 j (cnt值)
        block_dot_product = j * sum_of_b_block
        total_dot_product += block_dot_product

        # 更新累积的 sid 数量
        cumulative_freq_before += n_j

    return total_dot_product, N

# 确保 summary 数组不为空
if summary.size == 0:
    print("Error: Input summary data is empty. Cannot calculate Gini coefficient.")
else:
    # 调用函数计算简化点积和 N
    result_simplified_dot_product, N_sids = simplified_dot_product(summary)

    print(f"The simplified dot product (2 * sum(j * rank_j)) is: {result_simplified_dot_product}")
    print(f"Total number of unique SIDs (N) is: {N_sids}")

    # 计算总的 item 数量 (sum(j * n_j))
    # sum_j_times_nj = np.dot(summary[:, 0], summary[:, 1])
    # 这一步计算的是所有 item_id 的总数，即 sum(cnt * cnt_freq)
    # 例如：(1*100) + (2*50) + (3*20) ...
    sum_j_times_nj = np.dot(summary[:, 0], summary[:, 1])

    # 确保分母不为零
    if N_sids == 0 or sum_j_times_nj == 0:
        print("Error: Denominator for Gini calculation is zero. Cannot calculate Gini coefficient.")
    else:
        # 基尼系数的计算公式
        gini = (N_sids + 1) / N_sids - (2 * result_simplified_dot_product / (N_sids * sum_j_times_nj))
        print(f"The Gini Coefficient is: {gini}")

```

### 2.2 运行步骤 (Execution Steps)

1. **执行 SQL:** 按照 1.1 节中的顺序执行 SQL 查询，生成 `leaf_sid_cnt_freq` 表。
2. **导出数据:** 将 `leaf_sid_cnt_freq` 表的数据导出到名为 `row_summary.txt` 的文件中。确保文件内容格式为 `cnt$$||$$cnt_freq`，例如：
   ```
   1$$||$$100
   2$$||$$50
   3$$||$$20
   ...
   ```
3. **运行 Python 脚本:** 将上述 Python 代码保存为 `calculate_gini.py`，然后在终端中运行：
   ```bash
   python calculate_gini.py
   ```
4. **查看结果:** 脚本将输出计算得到的简化点积和基尼系数。

---
