# 1.生成5种泛函的吸附能文件，先遍历吸附质，后遍历表面

# import numpy as np
# import matplotlib.pyplot as plt
# import ase.db
# from pymatgen import Element

# db = ase.db.connect('adsorption.db')
# db_surf = ase.db.connect('surfaces.db')

# adsorbates = ['OH','CH','NO','CO','N2','N','O','H']
# num_list = [21,22,23,24,25,26,27,28,29,39,40,41,42,44,45,46,47,72,73,74,75,76,77,78,79]


# te1 = open('LDA_adsorp.txt','w')
# te2 = open('PBE_adsorp.txt','w')
# te3 = open('RPBE_adsorp.txt','w')
# te4 = open('BEEFvdW_adsorp.txt','w')
# te5 = open('RPA_adsorp.txt','w')

# for i in num_list:
#     ele = str(Element.from_Z(i))
#     for adsorbate in adsorbates:
#         rows = db.select(adsorbate=adsorbate)
#         for row in rows:
#             if row.symbols[0] == ele:
#                 print(adsorbate+","+ele+","+str(row.LDA_adsorp),file=te1)
#                 print(adsorbate+","+ele+","+str(row.PBE_adsorp),file=te2)
#                 print(adsorbate+","+ele+","+str(row.RPBE_adsorp),file=te3)
#                 print(adsorbate+","+ele+","+str(row.BEEFvdW_adsorp),file=te4)
#                 print(adsorbate+","+ele+","+str(row.RPA_adsorp),file=te5)
                
# te1.close()
# te2.close()
# te3.close()
# te4.close()
# te5.close()



# 2.分别生成单原子和双原子吸附质的吸附能，只用RPA泛函(3*25)

# import numpy as np
# import matplotlib.pyplot as plt
# import ase.db
# from pymatgen import Element

# db = ase.db.connect('adsorption.db')
# db_surf = ase.db.connect('surfaces.db')

# adsorbates = ['N','O','H']
# double_adsorbates = ['OH','NO','CO']
# num_list = [21,22,23,24,25,26,27,28,29,39,40,41,42,44,45,46,47,72,73,74,75,76,77,78,79]

# energy_singe = []
# energy_double = []

# te_single = open('RPA_adsorp_single.txt','w')

# for i in num_list:
#     ele = str(Element.from_Z(i))
#     for adsorbate in adsorbates:
#         rows = db.select(adsorbate=adsorbate)
#         for row in rows:
#             if row.symbols[0] == ele:
#                 print(adsorbate+","+ele+","+str(row.RPA_adsorp),file=te_single)
#                 energy_singe.append(row.RPA_adsorp)
# te_single.close()

# te_double = open('RPA_adsorp_double.txt','w')

# for i in num_list:
#     ele = str(Element.from_Z(i))
#     for double_adsorbate in double_adsorbates:
#         rows = db.select(adsorbate=double_adsorbate)
#         for row in rows:
#             if row.symbols[0] == ele:
#                 print(double_adsorbate+","+ele+","+str(row.RPA_adsorp),file=te_double)
#                 energy_double.append(row.RPA_adsorp)
# te_double.close()



# 3.1.生成模型输入X和y，先考虑单原子吸附，RPA泛函，7种feature，共75组数据（第一种输入）
# import numpy as np
# import pandas as pd
# a = pd.read_excel("EP.xlsx",index_col="Element")
# X = []
# y = []

# aa = open('RPA_adsorp_single.txt','r')       
# nn=aa.readlines()

# for i in range(len(nn)):
#     mol= str(nn[i].replace(","," ").split()[0])
#     sur= str(nn[i].replace(","," ").split()[1])
#     en = float(nn[i].replace(","," ").split()[2])
#     xi = []
#     yi = en
#     xi.append(a.loc[mol,"R"])
#     xi.append(a.loc[mol,"IE"])
#     xi.append(a.loc[mol,"FUS"])
#     xi.append(a.loc[sur,"G"])
#     xi.append(a.loc[sur,"FUS"])
#     xi.append(a.loc[sur,"Rs"])
#     xi.append(a.loc[sur,"surface energy"])
#     X.append(xi)
#     y.append(yi)

# X = np.array(X)
# y = np.array(y)

# #best score:0.9141482010102042

# 3.2.生成模型输入X和y，先考虑单原子吸附，RPA泛函，9种feature,新加入的feature为d带中心和功函数,共75组数据（第一种输入）
# import numpy as np
# import pandas as pd
# a = pd.read_excel("EP.xlsx",index_col="Element")
# X = []
# y = []

# aa = open('RPA_adsorp_single.txt','r')       
# nn=aa.readlines()

# for i in range(len(nn)):
#     mol= str(nn[i].replace(","," ").split()[0])
#     sur= str(nn[i].replace(","," ").split()[1])
#     en = float(nn[i].replace(","," ").split()[2])
#     xi = []
#     yi = en
#     xi.append(a.loc[mol,"R"])
#     xi.append(a.loc[mol,"IE"])
#     xi.append(a.loc[mol,"FUS"])
#     xi.append(a.loc[sur,"G"])
#     xi.append(a.loc[sur,"FUS"])
#     xi.append(a.loc[sur,"Rs"])
#     xi.append(a.loc[sur,"surface energy"])
#     xi.append(a.loc[sur,"d band center"])
#     xi.append(a.loc[sur,"work function"])
#     X.append(xi)
#     y.append(yi)

# X = np.array(X)
# y = np.array(y)

## best score:0.9143360490866725


# 3.3.生成模型X和y，考虑双原子吸附，7种feature,共75组数据
# import numpy as np
# import pandas as pd
# a = pd.read_excel("EP.xlsx",index_col="Element")
# X = []
# y = []

# aa = open('RPA_adsorp_double.txt','r')       
# nn=aa.readlines()

# for i in range(len(nn)):
#     sub= str(nn[i].replace(","," ").split()[0])
#     sur= str(nn[i].replace(","," ").split()[1])
#     en = float(nn[i].replace(","," ").split()[2])
#     if sub == 'OH':
#         mol = 'H'
#     elif sub == 'NO':
#         mol = 'N'
#     else:
#         mol = 'C'
#     xi = []
#     yi = en
#     xi.append(a.loc[mol,"G"])
#     xi.append(a.loc[mol,"AM"])
#     xi.append(a.loc[mol,"P"])
#     xi.append(a.loc[mol,"FUS"])
#     xi.append(a.loc[sur,"G"])
#     xi.append(a.loc[sur,"Rs"])
#     xi.append(a.loc[sur,"surface energy"])
#     X.append(xi)
#     y.append(yi)

# X = np.array(X)
# y = np.array(y)

### best score:0.6327624836290693

# 3.4.生成模型X和y，考虑双原子吸附，9种feature,加上d带中心和功函数，共75组数据
import numpy as np
import pandas as pd
a = pd.read_excel("EP.xlsx",index_col="Element")
X = []
y = []

aa = open('RPA_adsorp_double.txt','r')       
nn=aa.readlines()

for i in range(len(nn)):
    sub= str(nn[i].replace(","," ").split()[0])
    sur= str(nn[i].replace(","," ").split()[1])
    en = float(nn[i].replace(","," ").split()[2])
    if sub == 'OH':
        mol = 'H'
    elif sub == 'NO':
        mol = 'N'
    else:
        mol = 'C'
    xi = []
    yi = en
    xi.append(a.loc[mol,"G"])
    xi.append(a.loc[mol,"AM"])
    xi.append(a.loc[mol,"P"])
    xi.append(a.loc[mol,"FUS"])
    xi.append(a.loc[sur,"G"])
    xi.append(a.loc[sur,"Rs"])
    xi.append(a.loc[sur,"surface energy"])
    xi.append(a.loc[sur,"d band center"])
    xi.append(a.loc[sur,"work function"])
    X.append(xi)
    y.append(yi)

X = np.array(X)
y = np.array(y)

## best score:

# 4.构建KNN模型
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

train_ind, valid_ind = train_test_split(np.arange(X.shape[0]),
                                        test_size=0.2, shuffle=True,
                                        random_state=123)

pipe = make_pipeline(StandardScaler(),
                     KNeighborsRegressor())

params = {'kneighborsregressor__n_neighbors': [1, 3, 5],
          'kneighborsregressor__p': [1, 2]}

split = PredefinedHoldoutSplit(valid_indices=valid_ind)

grid = GridSearchCV(pipe,
                    param_grid=params,
                    cv=split)

grid.fit(X, y)
print(grid.best_score_)






















