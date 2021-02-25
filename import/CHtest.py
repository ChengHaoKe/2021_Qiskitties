import pandas as pd
import numpy as np


income0 = pd.read_csv('raw_data/median_hh_income2015.csv')
poverty0 = pd.read_csv('raw_data/percentage_below_poverty.csv')
over0 = pd.read_csv('raw_data/percentage_over25_complete_hs.csv')
race0 = pd.read_csv('raw_data/percentage_race_by_city.csv')
kill0 = pd.read_csv('raw_data/police_killings.csv')

kill0['Geographic_Area'] = kill0['state']
kill0 = kill0.rename(columns={'city': 'City'})
income0 = income0.rename(columns={'Geographic Area': 'Geographic_Area'})
poverty0 = poverty0.rename(columns={'Geographic Area': 'Geographic_Area'})
over0 = over0.rename(columns={'Geographic Area': 'Geographic_Area'})
race0 = race0.rename(columns={'Geographic area': 'Geographic_Area'})


dflist = [income0, poverty0, over0, race0, kill0]
keys = []
for k in dflist[:4]:
    keys.append(k[['Geographic_Area', 'City']])

kdf0 = pd.concat(keys)
kdf0 = kdf0.drop_duplicates()
# kdf0['City1'] = kdf0['City'].str.replace('city|borough|CDP|village|town', '')
# kdf0['City1'] = kdf0['City'].str.split(' ')
# kdf0['City2'] = kdf0['City1'].apply(lambda x: x[0])


alldf = kdf0
keys0 = ['Geographic_Area', 'City']
for i, k in enumerate(dflist[:4]):
    alldf = pd.merge(alldf, k, how='left', on=keys0)

alldf['City1'] = alldf['City'].str.split(' ')
alldf['City2'] = alldf['City1'].apply(lambda x: x[0])
kill0['City1'] = kill0['City'].str.split(' ')
kill0['City2'] = kill0['City1'].apply(lambda x: x[0])
alldf = pd.merge(alldf, kill0, how='left', on=['Geographic_Area', 'City2'])

alldf0 = alldf.dropna()


# ################################################ by state ###################################
income0['Median_Income'] = income0['Median Income'].astype(float)
income1 = income0.groupby('Geographic_Area', as_index=False).agg({'Median_Income': 'mean'})

poverty0['poverty_rate'] = poverty0['poverty_rate'].astype(float)
poverty1 = poverty0.groupby('Geographic_Area', as_index=False).agg({'poverty_rate': 'mean'})

over0['percent_completed_hs'] = over0['percent_completed_hs'].astype(float)
over1 = over0.groupby('Geographic_Area', as_index=False).agg({'percent_completed_hs': 'mean'})

racecol = ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']
for r in racecol:
    race0[r] = race0[r].str.replace(r"[a-zA-Z]|\(|\)", '', regex=True)
    race0[r] = race0[r].apply(lambda x: float(x) if x != '' else np.nan)

racedict = {r: 'mean' for r in racecol}
race1 = race0.groupby('Geographic_Area', as_index=False).agg(racedict)

# kill1 = kill0.groupby('Geographic_Area', as_index=False).agg({'Median_Income': 'mean'})
#
#
# alldf = kdf0
# keys0 = ['Geographic_Area', 'City']
# for i, k in enumerate(dflist[:4]):
#     alldf = pd.merge(alldf, k, how='left', on=keys0)

# ################################################ killing centered ###################################
kill0['citylist'] = kill0['City'].str.split(' ')
kill0['lencity'] = kill0['citylist'].apply(lambda x: len(x))
alldf['citylist'] = alldf['City'].str.split(' ')

# all1 = kill0[kill0['City'].isin(alldf['City'])]
all0 = kill0[kill0['City'].apply(lambda x: alldf['City'].str.contains(x)).any(1)]

all2 = []
for k0 in range(len(kill0)):
    k1 = kill0.iloc[k0]
    a0 = alldf[alldf['Geographic_Area'] == k1['Geographic_Area']]
    for a1 in range(len(a0)):
        a2 = a0.iloc[a1]
        if k1['citylist'][:k1['lencity']] == a2['citylist'][:k1['lencity']]:
            ka0 = pd.concat([k1, a2], axis=0)
            ka0 = pd.DataFrame(ka0).T
            all2.append(ka0)

all3 = pd.concat(all2)
all3 = all3.reset_index(drop=True)

# duplicates: only 2268 ids in all3
dup0 = all3[all3.duplicated(['id'])]

all3.to_csv('int_data/CH_data.csv', index=False, encoding='utf_8_sig')
