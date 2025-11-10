import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# Configs
base_dir = '/your_path/EnDeep4mC/data/4mC'
fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli2', '4mC_G.subterraneus', '4mC_G.pickeringii']
#fasta_dataset_names = ['4mC_G.subterraneus']
file_types = ['train_pos.txt', 'train_neg.txt', 'test_pos.txt', 'test_neg.txt']

# Dictionary for storing GC content
species_gc = {species: [] for species in fasta_dataset_names}

# Traverse all species and files
for species in fasta_dataset_names:
    species_dir = os.path.join(base_dir, species)
    if not os.path.exists(species_dir):
        print(f"Warning: Directory not found - {species_dir}")
        continue
    
    for file_type in file_types:
        file_path = os.path.join(species_dir, file_type)
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue

        # Read FASTA file and calculate GC content
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                seq = str(record.seq).upper()
                gc = gc_fraction(seq) * 100  # Convert to percentage
                species_gc[species].append(gc)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

# Data statistics and visualization
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid", palette="muted")

# Prepare drawing data
plot_data = []
for species, gc_values in species_gc.items():
    if gc_values:  # Skip empty dataset
        short_name = species.split('_')[-1]  # Extract species abbreviations
        plot_data.append(pd.DataFrame({
            'Species': [short_name]*len(gc_values),
            'GC%': gc_values
        }))

# Merge data and draw box plots
if plot_data:
    df = pd.concat(plot_data)

    ax = sns.boxplot(x='Species', y='GC%', data=df, showfliers=False)
    plt.title('GC Content Distribution Across Species', fontsize=14)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('GC Content (%)', fontsize=12)

    medians = df.groupby('Species')['GC%'].median().values
    for i, m in enumerate(medians):
        ax.text(i, m+1, f'Median: {m:.1f}%', 
                horizontalalignment='center',
                fontsize=9, color='darkblue')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('GC_content_comparison.png', dpi=300)
    plt.show()

    print("\nGC Content Statistics:")
    print(df.groupby('Species')['GC%'].describe())
else:
    print("Error: No valid GC content data found!")

gs_data = species_gc.get('4mC_G.subterraneus', [])
print(f"\nG.subterraneus samples: {len(gs_data)} (others: { {k: len(v) for k, v in species_gc.items()} })")