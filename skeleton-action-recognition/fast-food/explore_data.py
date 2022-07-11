import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', required=True)
args = parser.parse_args()

CSV_FILE = args.csv_file

data_summary = pd.read_csv(CSV_FILE)

categories = list(data_summary['category'].unique())
categories.remove(' ')

lengths = []
for _, i in data_summary.iterrows():
    lengths.append(eval(i['shape'])[1])

data_summary['lengths'] = lengths

# sns.scatterplot(data=data_summary, x='instance_id', y='lengths', hue='category')
# plt.show()

for c in categories:

    print(f"{c} : {data_summary.loc[data_summary['category'] == c]['lengths'].mean()}")

print(data_summary['category'].value_counts())