import json

def export_to_json(data, filename='output.json'):
    """Exporta los datos al formato JSON y los guarda en un archivo."""
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Datos exportados a {filename}")
