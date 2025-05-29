import pandas as pd
import io

def format_dropdown_options(raw_values, lowercase_words=None):
    if lowercase_words is None:
        lowercase_words = {"and"}

    def format_label(value):
        words = value.replace("-", " ").split()
        return " ".join([
            word.capitalize() if word.lower() not in lowercase_words else word.lower()
            for word in words
        ])

    # Generate labels and check for duplicates
    seen = {}
    options = []
    mapping = {}
    for raw in raw_values:
        label = format_label(raw)
        if label in seen:
            label += f" ({raw})"  # Ensure uniqueness
        seen[label] = True
        options.append(label)
        mapping[label] = raw

    return options, mapping


# function finds us the correct column and value to use as a condition based on the dropdown selection
def map_region_condition(region_selection):
    
    list_of_continents = ['Africa',
                          'Antarctica',
                          'Asia',
                          'Europe',
                          'North America',
                          'Oceania',
                          'South America']
    
    region_mapping = {
        'EU': {
            'column_name': 'eu',
            'column_value': True
        },
        'OECD': {
            'column_name': 'oecd',
            'column_value': True
        },
        'Non-OECD': {
            'column_name': 'oecd',
            'column_value': False
        },
        'UNFCCC Annex': {
            'column_name': 'unfccc_annex',
            'column_value': True
        },
        'UNFCCC Non-Annex': {
            'column_name': 'unfccc_annex',
            'column_value': False
        },
        'Global North': {
            'column_name': 'developed_un',
            'column_value': True
        },
        'Global South': {
            'column_name': 'developed_un',
            'column_value': False
        },
        'Developed Markets': {
            'column_name': 'em_finance',
            'column_value': False
        },
        'Emerging Markets': {
            'column_name': 'em_finance',
            'column_value': True
        }
    }

    if region_selection == 'Global':
        return None
    
    elif region_selection in list_of_continents:
        return {
            'column_name': 'continent',
            'column_value': region_selection
        }
    
    elif region_selection in region_mapping:
        return region_mapping[region_selection]

    else:
        return {
            'column_name': 'country_name',
            'column_value': region_selection
        }
    

def format_number_short(n):
    if abs(n) >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1e3:.0f}K"
    else:
        return f"{n:.0f}"
    

def create_excel_file(dataframes_dict):
    
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    output.seek(0)
    return output