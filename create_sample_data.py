import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os

# Set a random seed for reproducibility
np.random.seed(42)

def generate_sample_data(n_samples=500):
    """
    Generate synthetic mosquito genomic data with different features for classification.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic genomic data
    """
    # Define the species to include
    species = ['Anopheles_gambiae', 'Aedes_aegypti', 'Culex_pipiens', 'Anopheles_stephensi']
    n_species = len(species)
    
    # Assign roughly equal number of samples to each species
    samples_per_species = n_samples // n_species
    
    # Create labels
    labels = np.concatenate([np.full(samples_per_species, species[i]) for i in range(n_species)])
    
    # In case there's a remainder due to integer division
    if len(labels) < n_samples:
        remainder = n_samples - len(labels)
        labels = np.concatenate([labels, np.full(remainder, species[0])])
    
    # Shuffle the labels
    np.random.shuffle(labels)
    
    # Generate features
    # 1. Gene expression data (10 genes with different expression patterns per species)
    gene_expressions = np.zeros((n_samples, 10))
    
    # Define mean expression patterns for each species
    gene_means = {
        'Anopheles_gambiae': np.random.normal(5, 1, 10),
        'Aedes_aegypti': np.random.normal(7, 1, 10),
        'Culex_pipiens': np.random.normal(3, 1, 10),
        'Anopheles_stephensi': np.random.normal(6, 1, 10)
    }
    
    # Generate expressions for each sample based on its species
    for i, species_name in enumerate(labels):
        gene_expressions[i] = np.random.normal(gene_means[species_name], 1)
    
    # Scale gene expressions
    gene_expressions = scale(gene_expressions)
    
    # 2. SNP data (20 SNPs)
    snp_data = np.zeros((n_samples, 20))
    
    # Different SNP patterns for each species
    snp_probs = {
        'Anopheles_gambiae': np.random.uniform(0.2, 0.4, 20),
        'Aedes_aegypti': np.random.uniform(0.5, 0.7, 20),
        'Culex_pipiens': np.random.uniform(0.1, 0.3, 20),
        'Anopheles_stephensi': np.random.uniform(0.4, 0.6, 20)
    }
    
    # Generate SNP data (0, 1, or 2 representing homozygous reference, heterozygous, homozygous alternate)
    for i, species_name in enumerate(labels):
        for j in range(20):
            p = snp_probs[species_name][j]
            # Simplified model of genotypes
            probs = [(1-p)**2, 2*p*(1-p), p**2]  # Probabilities for 0, 1, 2
            snp_data[i, j] = np.random.choice([0, 1, 2], p=probs)
    
    # 3. Morphological characteristics (5 features)
    morphology = np.zeros((n_samples, 5))
    
    # Means for morphological features for each species
    morph_means = {
        'Anopheles_gambiae': [2.5, 0.8, 1.2, 3.3, 0.5],
        'Aedes_aegypti': [3.1, 0.6, 1.5, 2.8, 0.7],
        'Culex_pipiens': [2.8, 0.7, 1.0, 3.0, 0.4],
        'Anopheles_stephensi': [2.6, 0.9, 1.3, 3.2, 0.6]
    }
    
    # Generate morphological data
    for i, species_name in enumerate(labels):
        morphology[i] = np.random.normal(morph_means[species_name], 0.1)
    
    # Combine all features
    all_features = np.hstack((gene_expressions, snp_data, morphology))
    
    # Create column names
    columns = (
        [f'gene_expr_{i}' for i in range(1, 11)] + 
        [f'snp_{i}' for i in range(1, 21)] + 
        ['body_length', 'wing_width', 'proboscis_length', 'leg_length', 'thorax_width']
    )
    
    # Create the DataFrame
    df = pd.DataFrame(all_features, columns=columns)
    df['species'] = labels
    
    return df

def main():
    print("Generating synthetic mosquito genomic dataset...")
    df = generate_sample_data(n_samples=500)
    
    # Print dataset summary
    print(f"Dataset shape: {df.shape}")
    print("\nSpecies distribution:")
    print(df['species'].value_counts())
    
    # Save to CSV
    output_path = 'mosquito_genomic_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {os.path.abspath(output_path)}")
    print(f"First 5 rows of the dataset:")
    print(df.head())

if __name__ == "__main__":
    main()

