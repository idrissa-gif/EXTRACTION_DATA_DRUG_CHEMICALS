import pandas as pd
import json
from Bio import Entrez
import time

# --- CONFIGURATION ---
CSV_FILE = "cleaned_target_entities.csv"   # The file with PMIDs and Terms
OUTPUT_FILE = "test_set_with_text.json"
EMAIL = "idrissa@gmail.com"  # NCBI requires an email address
API_KEY = None                    # Optional: Add NCBI API Key if you have one for faster speed

def fetch_abstracts(pmids):
    """
    Takes a list of PMIDs and returns a dictionary: { '12345': 'Title. Abstract text...' }
    """
    Entrez.email = EMAIL
    if API_KEY: Entrez.api_key = API_KEY
    
    print(f"Fetching abstracts for {len(pmids)} PMIDs from PubMed...")
    
    pmid_to_text = {}
    
    # Process in batches of 100 to respect API limits
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records['PubmedArticle']:
                # Extract PMID
                pmid = str(article['MedlineCitation']['PMID'])
                
                # Extract Title and Abstract
                article_data = article['MedlineCitation']['Article']
                title = article_data.get('ArticleTitle', "")
                abstract_list = article_data.get('Abstract', {}).get('AbstractText', [])
                
                # Join abstract parts (some are lists)
                abstract_text = " ".join(abstract_list) if abstract_list else ""
                
                # Combine Title + Abstract (Standard input for NER models)
                full_text = f"{title} {abstract_text}".strip()
                
                if full_text:
                    pmid_to_text[pmid] = full_text
                    
            print(f"  Processed batch {i}-{i+len(batch)}...")
            time.sleep(0.5) # Be polite to the server
            
        except Exception as e:
            print(f"  Error fetching batch {i}: {e}")
            
    return pmid_to_text

def create_gold_standard():
    # 1. Load CSV
    # Assuming semicolon delimiter
    df = pd.read_csv(CSV_FILE, sep=';', dtype=str)
    
    # Filter only rows that are True Positive (if your file has that column)
    if 'True positive' in df.columns:
        df = df[df['True positive'] == '1']

    # 2. Get Unique PMIDs
    unique_pmids = df['PMID'].unique().tolist()
    
    # 3. Download Texts
    pmid_map = fetch_abstracts(unique_pmids)
    
    # 4. Group annotations by PMID
    # We want: Input = Text, Expected = List of ALL terms in that text
    final_data = []
    grouped = df.groupby('PMID')
    
    for pmid, group in grouped:
        pmid = str(pmid)
        
        # Only proceed if we successfully downloaded the text
        if pmid in pmid_map:
            text = pmid_map[pmid]
            
            # Collect all ground truth entities for this abstract
            expected_entities = []
            for _, row in group.iterrows():
                expected_entities.append({
                    "term": row['Term'],
                    "id": row['Concept ID'],
                    "type": row.get('Terminology', 'Unknown') # e.g. 'chebi' or 'atc'
                })
            
            # Structure for the Evaluation Script
            final_data.append({
                "pmid": pmid,
                "text": text,
                "expected_entities": expected_entities
            })
        else:
            print(f"Warning: Could not fetch text for PMID {pmid}. Skipping.")

    # 5. Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"\nSuccess! Created {len(final_data)} test cases in {OUTPUT_FILE}")

if __name__ == "__main__":
    create_gold_standard()