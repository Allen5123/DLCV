import pandas as pd
import sys
pred, ans = sys.argv[1], sys.argv[2] 
df1 = pd.read_csv(pred)
df2 = pd.read_csv(ans)
n = len(df1['id'])
cnt = 0
for i in range(n):
    if df1['id'][i] != df2['id'][i]:
        print(f"id error {df1['id'][i]} != {df2['id'][i]}")
        exit(-1)
    if df1['filename'][i] != df2['filename'][i]:
        print(f"filename error {df1['filename'][i]} != {df2['filename'][i]}")
        exit(-1)
    cnt += 1 if df1['label'][i] == df2['label'][i] else 0

cnt /= n
print(f"acc : {cnt:.6%}")