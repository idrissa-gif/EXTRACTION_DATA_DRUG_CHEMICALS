import pandas as pd
import os

CSV_FILE_PATH = 'Database2025_data.csv'
WP2_FOLDER_PATH = 'D2_3'
SIBILS_INDEX_PATH = 'pmc_CR_title_set.tsv'

TARGET_ONTOLOGIES = ['atc', 'chebi'] # 'drugbank','pubchemmesh', 'covocchemicals'

def load_and_filter_data(csv_path):
    try:
        # Load CSV using semicolon separator
        df = pd.read_csv(csv_path, sep=';')

        # Clean column names
        df.columns = df.columns.str.strip()

        # Normalize terminology column
        df['Terminology'] = df['Terminology'].str.lower().str.strip()

        # Convert PMID to integer (drop invalid / missing)
        df['PMID'] = pd.to_numeric(df['PMID'], errors='coerce')
        df = df.dropna(subset=['PMID'])
        df['PMID'] = df['PMID'].astype(int)


        filtered_df = df[df['Terminology'].isin(TARGET_ONTOLOGIES)]

        print(f"Loaded {len(df)} rows. Retained {len(filtered_df)} rows after filtering.")
        return filtered_df

    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame()


def load_sibils_index(index_path):
    """
    Parses the TSV file correctly.
    Splits the line by tab '\t' and takes the first element (the ID).
    """
    if not os.path.exists(index_path):
        print(f"Warning: Sibils index file not found at {index_path}")
        return set()

    sibils_ids = set()
    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # FIX: Split by tab and take the first part (The ID)
                parts = line.split('\t')
                # Add the ID to the set
                sibils_ids.add(parts[0].strip())

    return sibils_ids


def locate_articles(unique_pmids, wp2_path, sibils_index):
    results = []

    # DEBUG: Print sample to verify what we are comparing
    print("\n--- DEBUG: ID Format Check ---")
    if len(sibils_index) > 0:
        print(f"Sample ID from Sibils Index: {list(sibils_index)[0]}")
    else:
        print("Sibils Index is empty.")
    print(f"Sample ID from CSV (PMID): {unique_pmids[0] if len(unique_pmids)>0 else 'None'}")
    print("------------------------------\n")

    for pmid_int in unique_pmids:
        status = "Not Found"
        location = "None"

        # 1. Check WP2 Folder (D2_3)
        # Assuming filename is just the number: "973458.txt"
        wp2_file_path = os.path.join(wp2_path, f"{pmid_int}.txt")

        if os.path.exists(wp2_file_path):
            status = "Found"
            location = "WP2 Deliverables (D2_3)"

        # 2. Check Sibils Index
        else:
            # We must check various formats because of the mismatch
            # Check 1: Direct Match (Unlikely if one is Int and other is "PMC...")
            # Check 2: Try adding "PMC" prefix just in case PMID and PMC numbers happened to be same (rare)

            pmid_str = str(pmid_int)
            pmc_str = f"PMC{pmid_int}"

            if pmid_str in sibils_index:
                status = "Found"
                location = "Sibils (Direct Match)"
            elif pmc_str in sibils_index:
                status = "Found"
                location = "Sibils (PMC Prefix Match)"
            # Note: This does not fix the PMID != PMCID numbering difference.

        results.append({
            'PMID': pmid_int,
            'Status': status,
            'Location': location
        })

    return pd.DataFrame(results)


def main():
    filtered_data = load_and_filter_data(CSV_FILE_PATH)

    if filtered_data.empty:
        print("No data to process.")
        return

    unique_pmids = filtered_data['PMID'].unique()

    # Load and Parse the TSV correctly
    sibils_ids = load_sibils_index(SIBILS_INDEX_PATH)
    print(f"Loaded {len(sibils_ids)} IDs from Sibils index.")

    location_report = locate_articles(
        unique_pmids,
        WP2_FOLDER_PATH,
        sibils_ids
    )

    print("\n--- Article Location Report ---")
    # Print the first 10 results to check
    print(location_report.head(10))

    # Check if we actually found any in Sibils
    sibils_found = location_report[location_report['Location'].str.contains("Sibils")]
    print(f"\nTotal found in Sibils: {len(sibils_found)}")

    filtered_data.to_csv(
        'cleaned_target_entities.csv',
        index=False,
        sep=';'
    )

    print("\nSaved filtered entities to 'cleaned_target_entities.csv'")


if __name__ == "__main__":
    main()