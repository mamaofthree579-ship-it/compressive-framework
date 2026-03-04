import numpy as np
import pandas as pd

np.random.seed(2)
n=36
grp = np.repeat([1,2,3], n//3)
gut = np.random.uniform(1,7,n)
entropy = 0.9 - 0.1*gut + np.random.normal(0,0.1,n)
entropy[grp!=3] += np.random.normal(0.3,0.05,(grp!=3).sum())
df = pd.DataFrame({'grp':grp,'gut':gut,'entropy':entropy})
for m in [1,2,3]:
    sub = df[df.grp==m]
    if len(sub)>1:
        r = sub.gut.corr(sub.entropy)
        print(m, r)
