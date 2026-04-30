import csv
import xml.etree.ElementTree as ET
import os

def parse_obo(filepath):
    """Generic OBO parser for ChEBI and COVOC."""
    terms = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return terms

    current_term = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            # COVOC/ChEBI usually use [Term], but some OBOs use [Typedef] or [Instance]
            if line.startswith('[') and line.endswith(']'):
                if "id" in current_term and "name" in current_term:
                    terms.append(current_term)
                current_term = {}
            elif line.startswith('id:'):
                current_term['id'] = line.split('id:')[1].strip()
            elif line.startswith('name:'):
                current_term['name'] = line.split('name:')[1].strip()

        if "id" in current_term:
            terms.append(current_term)
    return terms

def parse_mesh_xml(filepath):
    """Memory-efficient MeSH XML parser."""
    terms = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return terms

    try:
        context = ET.iterparse(filepath, events=('end',))
        for event, elem in context:
            if elem.tag == 'DescriptorRecord':
                name_elem = elem.find('.//DescriptorName/String')
                ui_elem = elem.find('DescriptorUI')
                if name_elem is not None and ui_elem is not None:
                    terms.append({'id': ui_elem.text, 'name': name_elem.text})
                elem.clear()
    except Exception as e:
        print(f"Error parsing MeSH XML: {e}")
    return terms

def parse_atc_csv(filepath):
    """Specific parser for ATC.csv based on your provided headers."""
    terms = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return terms

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        # Based on your data, it uses standard commas
        reader = csv.DictReader(f)
        for row in reader:
            # Your headers: 'Class ID' and 'Preferred Label'
            raw_id = row.get('Class ID', '')
            name = row.get('Preferred Label', '')

            if raw_id and name:
                # Extract code from URL (http://.../ATC/D11AX06 -> D11AX06)
                clean_id = raw_id.split('/')[-1]
                terms.append({'id': clean_id, 'name': name})
    return terms

def main():
    print("--- Starting Extraction ---")

    # Updated file names to match your latest run
    data_sources = {
        'chebi': parse_obo('chebi.obo'),
        # 'covoc': parse_obo('covoc.obo'),
        # 'mesh': parse_mesh_xml('desc2026.xml'),
        # 'atc': parse_atc_csv('ATC.csv')
    }

    output_file = 'parsed_results.csv'

    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['PMID', 'Term', 'Terminology', 'Concept ID', 'True positive', 'False positive'])

        total_count = 0
        for source, terms in data_sources.items():
            print(f"Extracted {len(terms)} terms from {source}.")
            for item in terms:
                # Use 00000 as the PMID for all reached elements
                writer.writerow(['00000', item['name'], source, item['id'], '1', ''])
                total_count += 1

    print(f"\n--- Success! ---")
    print(f"Total rows saved: {total_count}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()