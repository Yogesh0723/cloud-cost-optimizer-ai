import pandas as pd
from faker import Faker
import random
import os

def generate_name_data_csv(num_samples=10000):
    """
    Generates a comprehensive dataset of names with an exact 50/50 split
    between the two formats, including explicit, hard-coded examples.

    Args:
        num_samples (int): The total number of random name samples to generate.

    Returns:
        None. Saves the generated data to a CSV file.
    """
    fake = Faker('en_GB')

    data = []

    titles = ['MR', 'MRS', 'MISS', 'MS', 'DR', 'PROF', '']
    suffixes = ['JR', 'SR', 'JUNIOR', 'SENIOR', '']

    # --- PART 1: Generate half the data in Format 1 (comma-separated) ---
    num_format1 = num_samples // 2
    print(f"Generating {num_format1} samples for Format 1 (comma-separated)...")
    for _ in range(num_format1):
        first_name = fake.first_name()
        last_name = fake.last_name()
        if random.random() > 0.8:
            first_name = f"{fake.first_name()}-{fake.first_name()}"
        if random.random() > 0.6:
            last_name = f"{fake.last_name()}-{fake.last_name()}"
        title = random.choice(titles) if random.random() > 0.5 else ''
        middle_name = fake.first_name() if random.random() > 0.5 else ''
        suffix = random.choice(suffixes) if random.random() > 0.6 else ''
        if middle_name and random.random() > 0.7:
            middle_name = f"{middle_name} {fake.first_name()}"
        
        line_parts = [last_name, title, first_name, middle_name, suffix]
        
        if random.random() > 0.5:
            line_parts[1] = ''; line_parts[3] = ''; line_parts[4] = ''
        elif random.random() > 0.7:
            line_parts[1] = ''; line_parts[3] = f"{fake.first_name()} {fake.first_name()}"; line_parts[4] = ''

        full_name_supplied = ", ".join(line_parts)

        data.append({
            'full_name_supplied': full_name_supplied,
            'Title': title,
            'First name': first_name,
            'Second name': middle_name,
            'Second Initial': middle_name.split()[0] if middle_name else '',
            'Surname': last_name,
            'Suffix': suffix,
            'Full name as supplied': full_name_supplied
        })

    # --- PART 2: Generate the other half in Format 2 (space-separated) ---
    num_format2 = num_samples - num_format1
    print(f"Generating {num_format2} samples for Format 2 (space-separated)...")
    for _ in range(num_format2):
        first_name = fake.first_name()
        last_name = fake.last_name()
        if random.random() > 0.8:
            first_name = f"{fake.first_name()}-{fake.first_name()}"
        if random.random() > 0.6:
            last_name = f"{fake.last_name()}-{fake.last_name()}"
        title = random.choice(titles) if random.random() > 0.5 else ''
        middle_name = fake.first_name() if random.random() > 0.5 else ''
        suffix = random.choice(suffixes) if random.random() > 0.6 else ''
        if middle_name and random.random() > 0.7:
            middle_name = f"{middle_name} {fake.first_name()}"

        line_parts = [t for t in [title, first_name, middle_name, last_name, suffix] if t]
        full_name_supplied = " ".join(line_parts)
        
        data.append({
            'full_name_supplied': full_name_supplied,
            'Title': title,
            'First name': first_name,
            'Second name': middle_name,
            'Second Initial': middle_name.split()[0] if middle_name else '',
            'Surname': last_name,
            'Suffix': suffix,
            'Full name as supplied': full_name_supplied
        })

    # --- PART 3: Add hard-coded examples (your special cases) ---
    explicit_examples = [
        {'text': "SEYSS, RAMON,TODD, ", 'Title': '', 'First name': 'RAMON', 'Second name': 'TODD', 'Second Initial': 'T', 'Surname': 'SEYSS', 'Suffix': ''},
        {'text': "MISS SARAH-ANNE RACHEL-ANNE, ", 'Title': 'MISS', 'First name': 'SARAH-ANNE', 'Second name': 'RACHEL-ANNE', 'Second Initial': 'R', 'Surname': '', 'Suffix': ''},
        {'text': 'MEALLY,MISS ANNE-KAT NELSON, ', 'Title': 'MISS', 'First name': 'ANNE-KAT', 'Second name': 'NELSON', 'Second Initial': 'N', 'Surname': 'MEALLY', 'Suffix': ''},
        {'text': 'DR ANDREW-JAMES GORRAY, ', 'Title': 'DR', 'First name': 'ANDREW-JAMES', 'Second name': 'GORRAY', 'Second Initial': 'G', 'Surname': '', 'Suffix': ''},
        {'text': 'MRS AMBS OTTE MEE CARRILL,', 'Title': 'MRS', 'First name': 'AMBS', 'Second name': 'OTTE MEE', 'Second Initial': 'O', 'Surname': 'CARRILL', 'Suffix': ''},
        {'text': 'MISS SUSAN-LEE MILTON,', 'Title': 'MISS', 'First name': 'SUSAN-LEE', 'Second name': 'MILTON', 'Second Initial': 'M', 'Surname': '', 'Suffix': ''},
    ]

    df = pd.DataFrame(data)
    df_explicit = pd.DataFrame(explicit_examples)
    data = pd.concat([df, df_explicit], ignore_index=True)
    
    os.makedirs('data_dir', exist_ok=True)
    file_path = os.path.join('data_dir', 'uk_names_dataset.csv')
    data.to_csv(file_path, index=False)
    
    print(f"\nGenerated {len(data)} total samples and saved to uk_names_dataset.csv")
    print("\nHere's a sample of the data:")
    print(data.head(10))

if __name__ == "__main__":
    generate_name_data_csv(num_samples=10000)