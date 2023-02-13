# %%
import pandas as pd

# %%
df = pd.read_csv(
    '/src/datasets/ESC-50 Dataset for Environmental Sound Classification/ESC-50-master/meta/esc50.csv')

print(len(df))
# %%
df_target = df.target
df_category = df.category
# %%
df_unique = pd.concat([df_target, df_category], axis=1)
# %%
df_unique = df_unique.drop_duplicates()

# %%
df_unique.insert(2, 'keep', 'true')
# %%
df_unique.sort_values('target', inplace=True)
# %%
df_unique.to_csv('./meta.csv', index=False)
