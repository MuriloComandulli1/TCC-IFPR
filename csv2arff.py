import pandas as pd
from sklearn.preprocessing import LabelEncoder

filename = 'features_earlyfusion_Test2'

# Carrega o CSV
csv_file = filename + '.csv'
df = pd.read_csv(csv_file)

# Codifica os rótulos (assumindo que a coluna de rótulo se chama 'label')
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Exibe o mapeamento dos rótulos
print("Mapeamento dos rótulos:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Função para salvar em formato ARFF
def save_to_arff(df, arff_file, relation_name="image_features"):
    with open(arff_file, 'w') as f:
        f.write(f"@RELATION {relation_name}\n\n")
        
        for column in df.columns:
            if column == 'label':
                # Especifica os valores nominais como 0 e 1
                f.write(f"@ATTRIBUTE {column} {{0,1}}\n")
            else:
                f.write(f"@ATTRIBUTE {column} REAL\n")
        
        f.write("\n@DATA\n")
        for _, row in df.iterrows():
            # Força o valor da coluna 'label' para ser 0 ou 1
            row_data = [str(int(val)) if column == 'label' else str(val) for column, val in zip(df.columns, row.values)]
            row_line = ','.join(row_data)
            f.write(f"{row_line}\n")

# Salva em formato ARFF
arff_file = filename + '.arff'
save_to_arff(df, arff_file)

print(f"Arquivo ARFF salvo como {arff_file}")
