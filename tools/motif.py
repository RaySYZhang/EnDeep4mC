import os
import subprocess
from Bio import SeqIO

# 配置参数
base_dir = '/your_path/EnDeep4mC/data/4mC'
meme_path = '//your_path/meme/bin/meme'  # MEME installation path
output_dir = './meme_results'
#fasta_dataset_names = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster', '4mC_E.coli2', '4mC_G.subterraneus','4mC_G.pickeringii']
fasta_dataset_names = ['4mC_G.subterraneus']

def prepare_fasta(species):
    """Merge positive sample sequences and format them as MEME inputs"""
    input_files = [
        os.path.join(base_dir, species, 'train_pos.txt'),
        os.path.join(base_dir, species, 'test_pos.txt')
    ]
    output_file = os.path.join(output_dir, f"{species}_pos.fasta")
    
    # Merge sequences and standardize formats
    with open(output_file, 'w') as fout:
        for fin_path in input_files:
            if not os.path.exists(fin_path):
                print(f"Warning: Missing {fin_path}")
                continue
            for record in SeqIO.parse(fin_path, 'fasta'):
                seq = str(record.seq).upper()  # 统一大写
                fout.write(f">{record.id}\n{seq}\n")
    return output_file

def run_meme(input_fasta, species):
    """Run MEME and save the results"""
    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)
    
    cmd = [
        meme_path,
        input_fasta,
        '-dna',
        '-mod', 'zoops',    # Allow motif to appear repeatedly
        '-nmotifs', '5',     # Find the top 5 significant motifs
        '-minw', '6',       # Minimum motif length
        '-maxw', '12',       # Maximum motif length
        '-revcomp',          # Consider dual chain
        '-oc', species_dir
    ]
    
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"Success: MEME analysis for {species}")
    except subprocess.CalledProcessError as e:
        print(f"Error in {species}: {e.stderr.decode()}")

def compare_motifs(target_species='4mC_G.subterraneus'):
    """Compare the motif differences between the target species and other species"""
    from glob import glob
    
    # Collect all motif files
    motif_files = {}
    for species in fasta_dataset_names:
        meme_out = os.path.join(output_dir, species, 'meme.txt')
        if os.path.exists(meme_out):
            motif_files[species] = meme_out
    
    # Using Tomtom for motif comparison
    target_motif = motif_files.get(target_species)
    if not target_motif:
        print("Target species motif not found!")
        return
    
    for comp_species, comp_file in motif_files.items():
        if comp_species == target_species:
            continue
        
        output_dir_comp = os.path.join(output_dir, f"vs_{comp_species.split('_')[-1]}")
        cmd = [
            os.path.join(os.path.dirname(meme_path), 'tomtom'),
            '-oc', output_dir_comp,
            '-thresh', '0.1',  # Loose threshold
            target_motif,
            comp_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Comparison with {comp_species} completed")
        except Exception as e:
            print(f"Tomtom failed for {comp_species}: {str(e)}")

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each species
    for species in fasta_dataset_names:
        fasta_file = prepare_fasta(species)
        if os.path.getsize(fasta_file) > 0:  # Check non empty files
            run_meme(fasta_file, species)
        else:
            print(f"Skipping empty file: {fasta_file}")
    
    # Perform cross species motif comparison
    compare_motifs()