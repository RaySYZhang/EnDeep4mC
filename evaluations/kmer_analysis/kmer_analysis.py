import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import fisher_exact
from concurrent.futures import ProcessPoolExecutor
from Bio import SeqIO

# log congfigs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kmer_analysis.log'),
        logging.StreamHandler()
    ]
)

class KMerAnalyzer:
    def __init__(self, base_dir, output_dir, k_values=(3,5), threads=8, whitelist=None):
        """
        Initialize analyzer
        :param base_dir: root directory containing data of species
        :param output_dir: output directory
        :param k_values: the analysis range of k values (start, end)
        :param threads: number of parallel threads
        :param whitelist: white list for species
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.k_start, self.k_end = k_values
        self.threads = threads
        self.whitelist = set(whitelist) if whitelist else None
        self.species_class = self._load_species_classification()
        self.non_fasta_species = {  # Non FASTA format species
            "4mC_C.equisetifolia", "4mC_E.coli",
            "4mC_F.vesca", "4mC_S.cerevisiae", "4mC_Tolypocladium"
        }
        
        # create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    def _load_species_classification(self):
        """Loading species classification information"""
        return {
            # eukaryotes
            "A.thaliana": "Plant", "A.thaliana2": "Plant",
            "C.equisetifolia": "Plant", "F.vesca": "Plant", "R.chinensis": "Plant", 
            "C.elegans": "Animal", "C.elegans2": "Animal",
            "D.melanogaster": "Animal", "D.melanogaster2": "Animal",
            # prokaryotes
            "E.coli": "Microorganism", "E.coli2": "Microorganism",
            "G.subterraneus": "Microorganism", "G.subterraneus2": "Microorganism",
            "G.pickeringii": "Microorganism", "G.pickeringii2": "Microorganism",
            "S.cerevisiae": "Microorganism", "Tolypocladium": "Microorganism"
        }

    def _read_sequences(self, species, seq_type):
        """
        Enhanced sequence reading, supporting FASTA/non FASTA formats
        :param species: dataset names of species (e.g. "4mC_E.coli")
        :param seq_type: 'pos' or 'neg'
        :return: sequence list
        """
        seqs = []
        species_dir = os.path.join(self.base_dir, species)
        
        for fname in [f'train_{seq_type}.txt', f'test_{seq_type}.txt']:
            fpath = os.path.join(species_dir, fname)
            if not os.path.exists(fpath):
                logging.warning(f"File does not exist: {fpath}")
                continue
                
            try:
                # determine whether it is in FASTA format
                is_fasta = False
                if species not in self.non_fasta_species:
                    with open(fpath) as f:
                        first_line = f.readline().strip()
                        is_fasta = first_line.startswith(">")
                    f.seek(0)  # reset file pointer

                # handling based on individual situations
                if is_fasta:
                    records = list(SeqIO.parse(fpath, "fasta"))
                    valid = 0
                    for record in records:
                        seq = str(record.seq).upper()
                        if len(seq) >= 6:
                            seqs.append(seq)
                            valid += 1
                    logging.info(f"{species} {fname}: Analyze FASTA sequences and obtain {valid} valid sequences")
                else:
                    with open(fpath) as f:
                        line_count = 0
                        for line in f:
                            seq = line.strip().upper()
                            if len(seq) >= 6 and set(seq).issubset({'A','T','C','G','N'}):
                                seqs.append(seq)
                                line_count += 1
                    logging.info(f"{species} {fname}: Read plain text and obtain {line_count} valid sequences")
                    
            except Exception as e:
                logging.error(f"Failed to read {fpath} : {str(e)}")
                continue
        
        logging.info(f"{species} {seq_type} Total number of sequences: {len(seqs)}")
        return seqs

    def _get_kmer_features(self, seqs, k):
        """Return original counts"""
        counter = defaultdict(int)
        for seq in seqs:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if 'N' not in kmer:
                    counter[kmer] += 1
        return counter

    def process_species(self, species):
        """
        Processing k-mer analysis of individual species
        :return: dictionary containing positive and negative sample kmer frequencies
        """
        logging.info(f"Start processing species: {species}")
        
        # Obtain species classification
        species_key = species.split('4mC_')[-1]
        group = self.species_class.get(species_key, 'Unknown')
        logging.info(f"{species} is divided into: {group}")
        
        # Read sequence
        pos_seqs = self._read_sequences(species, 'pos')
        neg_seqs = self._read_sequences(species, 'neg')
        
        # Empty data check
        if not pos_seqs and not neg_seqs:
            logging.error(f"{species} has no valid sequences, skip processing")
            return None
            
        # Calculate k-mer frequency
        results = {'species': species, 'group': group}
        
        for k in range(self.k_start, self.k_end+1):
            pos_kmers = self._get_kmer_features(pos_seqs, k)
            neg_kmers = self._get_kmer_features(neg_seqs, k)
            
            # Record statistical information
            results[f'k{k}_pos'] = pos_kmers
            results[f'k{k}_neg'] = neg_kmers
            logging.info(f"{species} k={k}: Positive samples have {len(pos_kmers)} kmer, while negtive ones have {len(neg_kmers)} kmer")
        
        return results

    def _merge_group_kmers(self, all_results):
        """Merge original counts"""
        merged = defaultdict(lambda: defaultdict(lambda: {'pos': defaultdict(int), 'neg': defaultdict(int)}))
        
        for result in all_results:
            if not result: continue
            group = result['group']
            for k in range(self.k_start, self.k_end+1):
                # Merge positive sample counts
                for kmer, count in result[f'k{k}_pos'].items():
                    merged[group][k]['pos'][kmer] += count
                # Merge negative sample counts
                for kmer, count in result[f'k{k}_neg'].items():
                    merged[group][k]['neg'][kmer] += count
                    
        return merged

    def _find_significant_kmers(self, group_kmers):
        """Perform Fisher's test using raw counts"""
        significant = []
        
        # Get the total count of all groups
        total_counts = defaultdict(lambda: defaultdict(lambda: {'pos': 0, 'neg': 0}))
        for group in group_kmers:
            for k in group_kmers[group]:
                total_counts[group][k]['pos'] = sum(group_kmers[group][k]['pos'].values())
                total_counts[group][k]['neg'] = sum(group_kmers[group][k]['neg'].values())

        for k in range(self.k_start, self.k_end+1):
            all_kmers = set()
            # Collect all kmers
            for group in group_kmers.values():
                all_kmers.update(group[k]['pos'].keys())
                all_kmers.update(group[k]['neg'].keys())
                
            for kmer in all_kmers:
                # Build a 2x2 contingency table
                contingency = []
                for group_type in ['Microorganism', 'Plant', 'Animal']:
                    if group_type not in group_kmers:
                        continue
                        
                    pos = group_kmers[group_type][k]['pos'].get(kmer, 0)
                    neg = group_kmers[group_type][k]['neg'].get(kmer, 0)
                    contingency.append([pos, neg])
                
                # Merge eukaryotic data
                if len(contingency) >= 2:
                    # Prokaryotes vs Eukaryotes (Plants+Animals)
                    prok = contingency[0]
                    euk = [sum(x) for x in zip(*contingency[1:])] if len(contingency) >1 else [0,0]
                    
                    # Perform Fisher's exact test
                    try:
                        _, pval = fisher_exact([prok, euk])
                    except Exception as e:
                        logging.error(f"Failed to inspect: {str(e)}")
                        continue
                    
                    # Calculate relative frequency for recording
                    prok_total = total_counts['Microorganism'][k]['pos'] + total_counts['Microorganism'][k]['neg']
                    euk_total = sum(total_counts[g][k]['pos']+total_counts[g][k]['neg'] for g in ['Plant','Animal'])
                    
                    record = {
                        'k': k,
                        'kmer': kmer,
                        'p_value': pval,
                        'prok_freq': (prok[0]/(prok_total or 1)) - (prok[1]/(prok_total or 1)),
                        'euk_freq': (sum(euk[::2])/(euk_total or 1)) - (sum(euk[1::2])/(euk_total or 1))
                    }
                    significant.append(record)
        
        return pd.DataFrame(significant)

    def visualize_results(self, df):
        """
        Generate visualization charts (enhance heatmap data validation)
        """
        if df.empty:
            logging.warning("No significant results, skip visualization")
            return
            
        # Filter significant results
        sig_df = df[df['p_value'] < 0.1].copy()
        
        # Add multiple validation correction
        from statsmodels.stats.multitest import multipletests
        if not sig_df.empty:
            _, pvals_corrected, _, _ = multipletests(sig_df['p_value'], method='fdr_bh')
            sig_df['p_adjusted'] = pvals_corrected
            sig_df = sig_df[sig_df['p_adjusted'] < 0.05]
        
        # Heatmap visualization
        try:
            plt.figure(figsize=(15, 10))
            heatmap_data = sig_df.pivot_table(
                index='kmer', 
                columns='k', 
                values='p_value', 
                aggfunc='min'
            )
            
            # data check
            if heatmap_data.empty:
                logging.warning("The heatmap data is empty, possible reason: different k values have no common kmer")
                return
                
            if heatmap_data.isnull().values.all():
                logging.warning("All p-values are NaN, check data validity")
                return
                
            # Handle zero values (to avoid log10 (0) errors)
            min_p = heatmap_data.min().min()
            if min_p <= 0:
                logging.warning(f"Invalid p-value {min_p} found, automatically corrected to 1e-10")
                heatmap_data = heatmap_data.clip(lower=1e-10)
            
            log_data = np.log10(heatmap_data)
            
            # Set a reasonable color range
            vmax = log_data.max().max()
            vmin = max(log_data.min().min(), -20)  # Prevent -∞
            
            sns.heatmap(
                log_data, 
                cmap='viridis_r',
                cbar_kws={'label': 'log10(p-value)'},
                vmin=vmin,
                vmax=vmax
            )

            plt.savefig(os.path.join(self.output_dir, 'figures', 'kmer_heatmap.png'), 
                    bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.error(f"Failed to generate heatmap: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_analysis(self):
        """Perform complete analysis process"""
        try:
            # Step 1: Collect data on all species
            all_species = [d for d in os.listdir(self.base_dir) if d.startswith('4mC_')]
            
            # Apply whitelist filtering
            if self.whitelist:
                original_count = len(all_species)
                species_dirs = [d for d in all_species if d in self.whitelist]
                valid_count = len(species_dirs)
                invalid = self.whitelist - set(species_dirs)
                
                logging.info(f"Whitelist effective: Enter {len(self.whitelist)} and find {valid_count} valid species")
                if invalid:
                    logging.warning(f"The following whitelist species were not found: {invalid}")
            else:
                species_dirs = all_species
                logging.info("Whitelist not enabled, processing all species")
            
            logging.info(f"The number of species to dealt with:({len(species_dirs)}):\n" + "\n".join(f"- {s}" for s in species_dirs))
            
            if not species_dirs:
                logging.error("There are no species available for analysis. Please check the input directory or whitelist settings")
                return

            # Parallel processing of species
            with ProcessPoolExecutor(max_workers=self.threads) as executor:
                all_results = list(executor.map(self.process_species, species_dirs))
            
            # Filter empty results
            valid_results = [res for res in all_results if res is not None]
            if not valid_results:
                logging.error("All species processing failed, please check the logs")
                return

            # Step 2: Merge data within the group
            group_kmers = self._merge_group_kmers(valid_results)
            
            # Step 3: Significance analysis
            sig_df = self._find_significant_kmers(group_kmers)
            
            if sig_df.empty:
                logging.warning("No significant k-mer was found")
                return
                
            # Save results
            sig_df.to_csv(os.path.join(self.output_dir, 'tables', 'significant_kmers.csv'), index=False)
            
            # Step 4: Visualization
            self.visualize_results(sig_df)
            
            logging.info("Analysis completed! The results are saved in: %s", self.output_dir)
            
        except Exception as e:
            logging.error(f"Abnormal termination of analysis process: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross group k-mer differential analysis')
    parser.add_argument('-i', '--input', required=True, help='Enter data directory path')
    parser.add_argument('-o', '--output', default='kmer_results', help='Output directory path')
    parser.add_argument('-k', '--kvalues', type=int, nargs=2, default=[3,5], 
                       help='Range of k values (e.g. 3 5 indicates analysis from 3-mer to 5-mer)')
    parser.add_argument('-t', '--threads', type=int, default=8, help='Number of parallel threads')
    parser.add_argument('-w', '--whitelist', 
                       help='List of whitelist species, separated by commas (e.g. "4mC_E.coli,4mC_A.thaliana")')
    
    args = parser.parse_args()
    
    # Processing whitelist
    whitelist = [s.strip() for s in args.whitelist.split(',')] if args.whitelist else None
    
    try:
        analyzer = KMerAnalyzer(
            base_dir=args.input,
            output_dir=args.output,
            k_values=tuple(args.kvalues),
            threads=args.threads,
            whitelist=whitelist
        )
        analyzer.run_analysis()
    except KeyboardInterrupt:
        logging.error("User interrupts execution")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)