import pickle
import pandas as pd
import numpy as np
import sys

def load_pickle(filename: str) -> dict:
    infile = open(filename, 'rb')
    dic = pickle.load(infile, encoding='latin1')
    infile.close()
    return dic

def load_df_from_pickle(filename):
    infile = open(filename, 'rb')
    df =  pickle.load(infile, encoding='latin1')
    infile.close()
    return df

def squash(df):
    df_dict = {}
    for x in range(df.shape[0]):
        serie = df.iloc[x,:]
        for index, value in serie.items():
            df_dict.update({f'{serie.name} {index}': f'{value}'})
            
    return pd.DataFrame.from_dict(df_dict, orient='index').T


args = sys.argv

if (len(args) != 4):
    print('Modo de uso: ')
    print('transform_dataset.py [input_dataset.pickle] [output_dataset.pickle] [df_transformado.csv]')
    sys.exit()

input_path =  args[1]
output_path =  args[2]
df_output_path = args[3]

input_dic = load_pickle(input_path)
output_dic = load_pickle(output_path)
input_df = pd.DataFrame.from_dict(input_dic, orient='index')
output_df = pd.DataFrame.from_dict(output_dic, orient='index')

print(f'Input shape: {input_df.shape}, Output shape: {output_df.shape}')

input_df['Load'] = input_df['Load'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['ConventionalGenerator'] = input_df['ConventionalGenerator'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['RenewableGenerator'] = input_df['RenewableGenerator'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['Switch'] = input_df['Switch'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['Line'] = input_df['Line'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['Transformer'] = input_df['Transformer'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)
input_df['Terminal'] = input_df['Terminal'].apply(lambda x: pd.DataFrame.from_dict(x, orient='index').T)

squash_input_df = input_df.copy(deep=True)
squash_input_df['Load'] = squash_input_df['Load'].apply(lambda x: squash(x))
squash_input_df['ConventionalGenerator'] = squash_input_df['ConventionalGenerator'].apply(lambda x: squash(x))
squash_input_df['RenewableGenerator'] = squash_input_df['RenewableGenerator'].apply(lambda x: squash(x))
squash_input_df['Switch'] = squash_input_df['Switch'].apply(lambda x: squash(x))
squash_input_df['Line'] = squash_input_df['Line'].apply(lambda x: squash(x))
squash_input_df['Transformer'] = squash_input_df['Transformer'].apply(lambda x: squash(x))
squash_input_df['Terminal'] = squash_input_df['Terminal'].apply(lambda x: squash(x))

squash_input_df['Load'] = squash_input_df['Load'].apply(lambda x: x.rename(mapper= lambda y: f'Load {y}', axis=1))
squash_input_df['ConventionalGenerator'] = squash_input_df['ConventionalGenerator'].apply(lambda x: x.rename(mapper= lambda y: f'ConventionalGenerator {y}', axis=1))
squash_input_df['RenewableGenerator'] = squash_input_df['RenewableGenerator'].apply(lambda x: x.rename(mapper= lambda y: f'RenewableGenerator {y}', axis=1))
squash_input_df['Switch'] = squash_input_df['Switch'].apply(lambda x: x.rename(mapper= lambda y: f'Switch {y}', axis=1))
squash_input_df['Line'] = squash_input_df['Line'].apply(lambda x: x.rename(mapper= lambda y: f'Line {y}', axis=1))
squash_input_df['Transformer'] = squash_input_df['Transformer'].apply(lambda x: x.rename(mapper= lambda y: f'Transformer {y}', axis=1))
squash_input_df['Terminal'] = squash_input_df['Terminal'].apply(lambda x: x.rename(mapper= lambda y: f'Terminal {y}', axis=1))

squash2 = squash_input_df[['Load', 'ConventionalGenerator', 'RenewableGenerator', 'Line', 'Terminal', 'Switch']]

total_dict = {}
for x in range(squash2.shape[0]):
    serie = squash2.iloc[x,:]
    inside_dict = {}
    for index, value in serie.items():
        for y in range(value.shape[0]):
            inside_serie = value.iloc[y,:]
            for in_index, in_value in inside_serie.items():
                inside_dict.update({f'{in_index}': f'{in_value}'})
    total_dict.update({f'{x}': inside_dict})
            
squash3 = pd.DataFrame.from_dict(total_dict, orient='index')

df_time = squash_input_df[['Year', 'Month', 'Day', 'Hour']]
df_time['Date'] = df_time.apply(lambda row: f"{row['Year']:.0f}-{row['Month']:.0f}-{row['Day']:.0f} {row['Hour']:.0f}:00:00", axis=1)

transformed_dataset = pd.concat([squash3.reset_index(), df_time['Date']], axis=1).drop(columns=['index'])

transformed_dataset = pd.concat([transformed_dataset, output_df], axis=1)

for col in transformed_dataset.columns:
    transformed_dataset[col] = pd.to_numeric(transformed_dataset[col], errors='ignore')

transformed_dataset.to_csv(df_output_path)

print(transformed_dataset.shape)