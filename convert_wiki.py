import pandas as pd
import sys


def excel_to_wikitable(excel_file, output_file="films_table.txt", sheet_name=0):
    # Charger le fichier Excel
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Colonnes fixes
    fixed_columns = ["Nom", "Date", "Type", "Mandat"]

    # Colonnes de notes = toutes celles qui ne sont pas fixes
    note_columns = [col for col in df.columns if col not in fixed_columns]

    # En-tête du tableau wiki
    wikitable = '{| class="wikitable sortable" style="text-align:center" width="100%"\n'
    wikitable += '|+Liste des Soirées\n'
    wikitable += '|-\n'
    wikitable += '! scope="col" | Nom\n'
    wikitable += '! scope="col" | Date\n'
    wikitable += '! scope="col" | Type\n'
    wikitable += '! scope="col" | Mandat\n'
    wikitable += '! scope="col" | Notes\n'
    
    mois_fr = ["janvier", "février", "mars", "avril", "mai", "juin",
           "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

    # Remplir les lignes du tableau
    for _, row in df.iterrows():
        # Récupérer et compresser les notes (supprimer NaN)
        notes = [str(row[col]) for col in note_columns if pd.notna(row[col])]
        notes_str = ", ".join(notes)

        date = pd.to_datetime(row["Date"])
        date_str = f"{date.day} {mois_fr[date.month - 1]} {date.year}"

        wikitable += '|-\n'
        wikitable += f'| {row["Nom"]} || {date_str} || {row["Type"]} || {row["Mandat"]} || {notes_str}\n'

    # Fermer le tableau
    wikitable += '|}'

    # Écrire dans un fichier texte
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(wikitable)

    print(f"✅ Table wiki enregistrée dans '{output_file}'")


# Exemple d’utilisation
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        sys.exit("Erreur: Specifie un unique chemin de fichier xlsx")
    else:
        xlsx_path = sys.argv[1]
    
    
    excel_to_wikitable(sys.argv[1], "table.txt")