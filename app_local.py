import os
import json
import chromadb
# Importuojame requests biblioteką, skirtą bendrauti su vietiniu API
import requests
# Importuojame dotenv biblioteką
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer

# 1. BENDRI NUSTATYMAI
# ---
# Įkeliame kintamuosius iš .env failo
load_dotenv()

# 2. Konfigūruojame vietinį LLM (Ollama)
# ---
LOCAL_LLM_URL = "http://localhost:11434/api/generate"
# Pakeiskite į modelio pavadinimą, kurį esate įkėlę į Ollama (pvz., 'llama3', 'mistral', 'phi3')
LOCAL_LLM_MODEL = "llama3"

# 3. CHROMADB NUSTATYMAI
# ---
DB_PATH = "./my_documents_db"  # Naudojame tą patį kelią, kuris buvo nustatytas vektorizavimo kode
INVOICE_COLLECTION_NAME = "invoices"
CONTRACT_COLLECTION_NAME = "contracts"

# 4. SERVERIO INICIALIZAVIMAS
# ---
app = Flask(__name__)

# Sentence-transformers modelis naudojamas ne šioje aplikacijoje, bet paliekame čia dėl konteksto
# sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# 5. CHROMADB KOLEKCIJŲ INICIALIZAVIMAS
# ---
print(f"Jungiamės prie ChromaDB ({DB_PATH})...")
client = chromadb.PersistentClient(path=DB_PATH)

# Nustatome NUORODAS į abi kolekcijas (sąskaitos ir sutartys)
try:
    invoice_collection = client.get_collection(name=INVOICE_COLLECTION_NAME)
    contract_collection = client.get_collection(name=CONTRACT_COLLECTION_NAME)
    print("ChromaDB kolekcijos sėkmingai rastos: sąskaitos ir sutartys.")
except Exception as e:
    print(
        f"KLAIDA: Nepavyko rasti ChromaDB kolekcijų. Patikrinkite, ar prieš tai įvykdėte vektorizavimo scenarijų. Klaida: {e}")
    exit()


@app.route('/')
def index():
    # Šiai aplikacijai reikia failo 'templates/index.html'
    return render_template('index.html')


def fetch_all_documents_from_collections():
    """
    Ištraukia ir sujungia visus tekstinius dokumentus (sąskaitas ir sutartis) iš abiejų ChromaDB kolekcijų.
    """
    all_context = []

    # 1. Ištraukiame sąskaitas
    try:
        invoice_ids = invoice_collection.get()['ids']
        if invoice_ids:
            invoice_docs = invoice_collection.get(ids=invoice_ids, include=['documents', 'metadatas'])
            for doc, meta in zip(invoice_docs['documents'], invoice_docs['metadatas']):
                # Pridedame dokumento tipo pavadinimą dėl aiškumo LLM modeliui
                all_context.append(f"[SĄSKAITA FAKTŪRA]: {doc}")
    except Exception as e:
        print(f"Įspėjimas: Nepavyko gauti sąskaitų duomenų: {e}")

    # 2. Ištraukiame sutartis
    try:
        contract_ids = contract_collection.get()['ids']
        if contract_ids:
            contract_docs = contract_collection.get(ids=contract_ids, include=['documents', 'metadatas'])
            for doc, meta in zip(contract_docs['documents'], contract_docs['metadatas']):
                all_context.append(f"[SUTARTIS]: {doc}")
    except Exception as e:
        print(f"Įspėjimas: Nepavyko gauti sutarčių duomenų: {e}")

    if not all_context:
        return None  # Grąžiname None, jei nerasta jokių dokumentų

    # Sujungiame visų dokumentų tekstą į vieną konteksto eilutę
    return "\n\n---\n\n".join(all_context)


@app.route('/ask', methods=['POST'])
def ask_local_llm():
    """
    Gauna užklausą, surenka kontekstą iš ChromaDB (abi kolekcijos) ir siunčia jį vietiniam LLM (Ollama).
    """
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Užklausa nerasta.'}), 400

        # Ištraukiame kontekstą iš abiejų kolekcijų
        context_text = fetch_all_documents_from_collections()

        if context_text is None:
            return jsonify({'response': 'Atsiprašau, duomenų bazėje nerasta jokių dokumentų (sąskaitų ar sutarčių).'})

        # PROMPT'AS VIETINIAM MODELIUI: Dabar nurodome, kad apdorojamos abi dokumentų rūšys
        full_prompt = f"""
        [INST]
        Jūs esate dirbtinio intelekto asistentas, specializuojantis verslo dokumentų (sąskaitų faktūrų ir sutarčių) analizėje.
        Atsakykite į vartotojo klausimą TIKSLIAI remdamiesi pateiktu kontekstu. Kontekste dokumentai yra pažymėti žymėmis [SĄSKAITA FAKTŪRA] arba [SUTARTIS].
        Būkite konkretus, išsamus ir nurodykite dokumentų tipus, kai atsakote.
        Nekurkite jokios informacijos, kurios nėra pateiktuose dokumentuose.
        Galutinį atsakymą PRIVALOTE pateikti lietuvių kalba.

        Kontekstas (įvairūs verslo dokumentai): 
        {context_text}

        Vartotojo klausimas: 
        {query}
        [/INST]
        """

        # API kvietimo į vietinį Ollama serverį konfigūracija
        ollama_payload = {
            "model": LOCAL_LLM_MODEL,
            "prompt": full_prompt,
            "stream": False  # Nustatome stream: False, kad gautume visą atsakymą iš karto
        }

        # Siunčiame POST užklausą į Ollama serverį
        ollama_response = requests.post(LOCAL_LLM_URL, json=ollama_payload)
        ollama_response.raise_for_status()

        response_data = ollama_response.json()
        response_text = response_data.get('response', 'Klaida: Nepavyko gauti atsakymo iš vietinio LLM.')

        # Grąžiname atsakymą į front-end
        return jsonify({'response': response_text})

    except requests.exceptions.ConnectionError:
        print(
            f"KLAIDA: Nepavyko prisijungti prie Ollama serverio. Patikrinkite, ar Ollama veikia ir ar modelis ({LOCAL_LLM_MODEL}) yra įkeltas.")
        return jsonify({'error': 'Klaida: Nepavyko prisijungti prie vietinio LLM serverio. Ar veikia Ollama?'}), 503
    except requests.exceptions.RequestException as e:
        print(f"Klaida siunčiant užklausą į Ollama: {e}")
        return jsonify({'error': f'Ollama API klaida: {str(e)}'}), 500
    except Exception as e:
        # Bendra klaidų ataskaita
        print(f"Įvyko klaida apdorojant užklausą: {e}")
        return jsonify({'error': f'Serverio klaida: {str(e)}'}), 500


if __name__ == '__main__':
    # Flask paleidimas
    app.run(debug=True)