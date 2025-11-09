import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# --- NUSTATYMAI ---

# JSON failų aplankai
INVOICES_FOLDER = "invoices"
CONTRACTS_FOLDER = "contracts"

# ChromaDB nustatymai
DB_PATH = "./my_documents_db"
INVOICE_COLLECTION_NAME = "invoices"
CONTRACT_COLLECTION_NAME = "contracts"

# 2. Įdėjimo modelio inicijavimas.
print("--- 1. ĮDĖJIMO MODELIO INICIAVIMAS ---")
print("⏳ Pradedamas 'Sentence-BERT' modelio (paraphrase-multilingual-mpnet-base-v2) įkėlimas/atsisiuntimas. Tai gali užtrukti kelias minutes...")
# Naudojamas tas pats daugeliakalbis modelis
try:
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("✅ Modelis įkeltas sėkmingai! (Apie 500 MB RAM)")
except Exception as e:
    print(f"❌ Klaida įkeliant modelį: {e}")
    exit()

# 3. ChromaDB duomenų bazės kliento ir kolekcijų inicijavimas.
print("\n--- 2. CHROMADB PRISIJUNGIMAS ---")
print(f"⏳ Jungiamės prie ChromaDB atminties saugyklos ({DB_PATH})...")
try:
    client = chromadb.PersistentClient(path=DB_PATH)
    # Inicijuojame dvi skirtingas kolekcijas
    invoice_collection = client.get_or_create_collection(name=INVOICE_COLLECTION_NAME)
    contract_collection = client = client.get_or_create_collection(name=CONTRACT_COLLECTION_NAME)
    print("✅ ChromaDB paruošta. Yra dvi kolekcijos: sąskaitos ir sutartys.")
    print(f"   Egzistuojančių sąskaitų skaičius: {invoice_collection.count()}")
    print(f"   Egzistuojančių sutarčių skaičius: {contract_collection.count()}")
except Exception as e:
    print(f"❌ Klaida jungiantis prie ChromaDB: {e}")
    exit()

# --- PAGALBINĖS FUNKCIJOS TEKSTO GENERAVIMUI ---

def create_invoice_text_representation(data: Dict[str, Any]) -> str:
    text_content = (
        f"PVM sąskaita faktūra Nr. {data.get('numeris', '')} išrašyta {data.get('data', '')}. "
        f"Pardavėjas: {data.get('pardavejas', {}).get('pavadinimas', '')}. "
        f"Gavėjas: {data.get('gavejas', {}).get('pavadinimas', '')}. "
        f"Bendra mokėtina suma: {data.get('sumos', {}).get('viso_su_pvm_eur', '0')} EUR. "
    )

    prekes = data.get('prekes', [])
    if prekes:
        text_content += "Prekės sąrašas: "
        item = prekes[0]
        text_content += (
            f"{item.get('pavadinimas', '')}, Kiekis: {item.get('kiekis_t', 'Nenurodyta')} t, "
            f"Viso: {item.get('viso_eur', '0')} EUR. "
        )
    return text_content.strip()


def create_contract_text_representation(data: Dict[str, Any]) -> str:
    """
    Sukuria tekstinę reprezentaciją iš sutarties JSON duomenų.
    """
    text_content = (
        f"Dokumento tipas: Sutartis, Nr. {data.get('numeris', '')}, Sudarymo data: {data.get('sudarymo_data', '')}. "
        f"Sutarties tipas: {data.get('sutarties_tipas', 'Nenurodyta')}. "
        f"Šalis A (Teikėjas/Pardavėjas): {data.get('salis_a', {}).get('pavadinimas', '')} (Įm. kodas: {data.get('salis_a', {}).get('imones_kodas', '')}). "
        f"Šalis B (Gavėjas/Pirkėjas): {data.get('salis_b', {}).get('pavadinimas', '')} (Įm. kodas: {data.get('salis_b', {}).get('imones_kodas', '')}). "
        f"Galiojimo terminas: {data.get('galiojimo_terminas', 'Nenurodyta')}. "
        f"Bendra vertė: {data.get('bendra_suma_eur', '0')} EUR. "
        f"Mokestis už paslaugas/prekes: {data.get('mokestis_uz_paslaugas', 'Nenurodyta')}. "
    )
    return text_content.strip()

# --- PAGRINDINĖ APDOROJIMO FUNKCIJA ---

def process_and_add_document(file_path: str, collection: chromadb.api.models.Collection, doc_type: str, text_generator_func):
    """
    Nuskaito JSON failą, vektorizuoja, įkelia į nurodytą ChromaDB kolekciją ir pašalina JSON failą.
    """
    file_name = os.path.basename(file_path)
    print(f"\n   ⚙️ Pradedamas apdoroti: {file_name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc_id = file_name.replace('.json', '')

        # Patikriname, ar dokumentas jau egzistuoja kolekcijoje
        try:
            if collection.get(ids=[doc_id])['ids']:
                print(f"   ⏭️ Dokumentas {doc_id} ({doc_type}) jau egzistuoja kolekcijoje, praleidžiamas.")
                # Nepašaliname, jei jis buvo tikrinamas anksčiau, bet neįkeltas
                return
        except KeyError:
            pass # Vykdome toliau, jei nerasta

        # Sukuriame unikalią tekstinę reprezentaciją
        text_content = text_generator_func(data)
        print(f"   📜 Teksto santrauka sugeneruota (Ilgis: {len(text_content)})")

        # Vektorizuojame tekstą
        print("   🧠 Generuojamas vektorius (Embedding)...")
        embedding = model.encode(text_content).tolist()
        print(f"   ✅ Vektorius sugeneruotas (Dydis: {len(embedding)})")

        # Įkeliame į ChromaDB
        collection.add(
            documents=[text_content],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{"json_data": json.dumps(data), "document_type": doc_type}]
        )
        print(f"   👍 Sėkmingai įkelta į ChromaDB: {file_name}")

        # Pašaliname sėkmingai įkeltą JSON failą
        os.remove(file_path)
        print(f"   🗑️ Originalus JSON failas '{file_name}' pašalintas.")

    except Exception as e:
        print(f"   ❌ Klaida apdorojant {file_name} ({doc_type}): {e}")

# --- MAIN FUNKCIJA ---

def main():
    """
    Pagrindinė funkcija, kuri apdoroja sąskaitas ir sutartis atskirai.
    """
    # 1. Apdorojame SĄSKAITAS FAKTŪRAS
    print("\n--- 3. PRADEDAMAS SĄSKAITŲ FAKTŪRŲ (Invoices) APDOROJIMAS ---")
    if not os.path.exists(INVOICES_FOLDER):
        print(f"Klaida: Aplankas '{INVOICES_FOLDER}' nerastas. Praleidžiama.")
    else:
        invoice_files = [f for f in os.listdir(INVOICES_FOLDER) if f.endswith('.json')]
        print(f"Rasti {len(invoice_files)} sąskaitų faktūrų JSON failai apdorojimui.")
        if not invoice_files:
            print(f"Aplanke '{INVOICES_FOLDER}' nerasta jokių naujų JSON failų.")
        else:
            for json_file in invoice_files:
                file_path = os.path.join(INVOICES_FOLDER, json_file)
                process_and_add_document(
                    file_path,
                    invoice_collection,
                    "invoice",
                    create_invoice_text_representation
                )

    # 2. Apdorojame SUTARTIS
    print("\n--- 4. PRADEDAMAS SUTARČIŲ (Contracts) APDOROJIMAS ---")
    if not os.path.exists(CONTRACTS_FOLDER):
        print(f"Klaida: Aplankas '{CONTRACTS_FOLDER}' nerastas. Praleidžiama.")
    else:
        contract_files = [f for f in os.listdir(CONTRACTS_FOLDER) if f.endswith('.json')]
        print(f"Rasti {len(contract_files)} sutarčių JSON failai apdorojimui.")
        if not contract_files:
            print(f"Aplanke '{CONTRACTS_FOLDER}' nerasta jokių naujų JSON failų.")
        else:
            for json_file in contract_files:
                file_path = os.path.join(CONTRACTS_FOLDER, json_file)
                process_and_add_document(
                    file_path,
                    contract_collection,
                    "contract",
                    create_contract_text_representation
                )

    print("\n" + "="*50)
    print("--- VISŲ FAILŲ APDOROJIMAS BAIGTAS. ---")
    print(f"Iš viso sąskaitų kolekcijoje: {invoice_collection.count()}")
    print(f"Iš viso sutarčių kolekcijoje: {contract_collection.count()}")
    print("="*50)

if __name__ == "__main__":
    main()