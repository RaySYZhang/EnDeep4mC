import subprocess
import os

def run_cd_hit(input_file, output_file, similarity=0.7):
    """
    Call CD-HIT for sequence clustering
    """
    # Set word_rength based on similarity
    if similarity >= 0.9:
        word_length = 5
    elif similarity >= 0.8:
        word_length = 4
    elif similarity >= 0.7:
        word_length = 3
    else:
        word_length = 2

    # Build CD-HIT command
    cmd = f"cd-hit -i {input_file} -o {output_file} -c {similarity} -n {word_length}"
    
    # Execute the command
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CD-HIT failed: {e}")
        raise

def generate_output_path(original_path):
    """
    Generate output path based on the original file path
    e.g.：/path/test_neg.txt -> /path/test_neg_cdhit.txt
    """
    base, ext = os.path.splitext(original_path)
    return f"{base}_cdhit{ext}"

def analyze_species(species_dir):
    """
    Handling dataset of a single species
    """
    target_files = [
        'test_neg.txt',
        'test_pos.txt',
        'train_neg.txt',
        'train_pos.txt'
    ]

    for filename in target_files:
        input_path = os.path.join(species_dir, filename)
        
        # Skip non-existent files
        if not os.path.exists(input_path):
            continue

        # Generate output path
        output_path = generate_output_path(input_path)
        clstr_path = output_path + ".clstr"

        # Run CD-HIT
        print(f"Processing: {input_path} -> {output_path}")
        run_cd_hit(input_path, output_path)

        # Statistical sequence information
        original_count = count_sequences(input_path)
        clustered_count = count_sequences(output_path)
        reduction_rate = (1 - clustered_count/original_count) * 100 if original_count else 0

        # Output statistical results
        print(f"Number of original sequences: {original_count}")
        print(f"Number of sequences after clustering: {clustered_count}")
        print(f"Sequence reduction rate: {reduction_rate:.1f}%\n")

def count_sequences(fasta_file):
    """Count the number of sequences in the FASTA file"""
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count

def process_all_species(root_dir):
    """
    Process datasets of all species
    """
    for species in os.listdir(root_dir):
        species_path = os.path.join(root_dir, species)
        if os.path.isdir(species_path):
            print(f"\n{'='*40}")
            print(f"Processing species: {species}")
            print(f"{'='*40}")
            analyze_species(species_path)

if __name__ == "__main__":
    data_root = "Projs/EnDeep4mC/data/4mC3"
    process_all_species(data_root)