import pandas as pd
df_1 = pd.read_csv('output_rag.csv')
df_2 = pd.read_csv('output_without_rag.csv')
result = []
result1 = []
result2 = []
for _, row in df_1.iterrows():
    idx = row['ID']
    answer = row['Answer']
    result1.append(answer)

for _, row in df_2.iterrows():
    idx = row['ID']
    answer = row['Answer']
    result2.append(answer)

for i in range(len(result1)):
    ans = result1[i] + result2[i]
    if ans >=2:
        result.append({"ID": i, "Answer": 1})
    else:
        result.append({"ID": i, "Answer": 0})

df = pd.DataFrame(result)
df.to_csv(f'output.csv', index=False)


