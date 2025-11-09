import os
import re
import json
import pdfplumber
from dotenv import load_dotenv
import google.generativeai as genai

# Įkeliame kintamuosius iš .env failo
load_dotenv()

# Konfiguracija
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Patikriname, ar raktas sėkmingai gautas
if not GOOGLE_API_KEY:
    print("Klaida: GOOGLE_API_KEY nerastas .env faile arba aplinkos kintamuosiuose. Patikrinkite .env failą.")
    exit()

# Konfigūruojame Gemini API
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Klaida konfigūruojant Gemini API: {e}")
    exit()

# Modelis, skirtas greitam duomenų ištraukimui
AI_MODEL = genai.GenerativeModel('gemini-2.5-flash')

# Aplankų nustatymas
# Dabar visus PDF laikysime viename aplanke "pdf_documents"
PDF_FOLDER_DOCUMENTS = "pdf_documents"
JSON_FOLDER_INVOICES = "invoices"
JSON_FOLDER_CONTRACTS = "contracts"

# Patikriname, ar egzistuoja išvesties aplankai ir juos sukuriame, jei reikia.
for folder in [JSON_FOLDER_INVOICES, JSON_FOLDER_CONTRACTS, PDF_FOLDER_DOCUMENTS]:
    if not os.path.exists(folder):
        print(f"Aplankas '{folder}' nerastas. Sukuriamas aplankas.")
        os.makedirs(folder)


def extract_text_from_pdf(pdf_path):
    """
    Ištraukia visą tekstą iš PDF failo.
    """
    text_content = ""
    try:
        # Padidintas x_tolerance/y_tolerance gali padėti su prastesnės kokybės PDF
        with pdfplumber.open(pdf_path) as pdf:
            # Paimame tik pirmąjį puslapį klasifikavimui (greičiau)
            text_content += pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=2) + "\n"
    except Exception as e:
        print(f"Klaida ištraukiant tekstą iš PDF '{pdf_path}': {e}")
    return text_content.strip()


def extract_full_text_from_pdf(pdf_path):
    """
    Ištraukia visą tekstą iš PDF failo.
    """
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text(x_tolerance=2, y_tolerance=2) + "\n"
    except Exception as e:
        print(f"Klaida ištraukiant tekstą iš PDF '{pdf_path}': {e}")
    return text_content.strip()


## RAGINIMAI (PROMPTS) lieka nepakitę, bet perkelti į pagrindinį kodo lygį dėl aiškumo.

def get_invoice_prompt(pdf_text):
    """
    Grąžina raginimą sąskaitos faktūros duomenų ištraukimui.
    """
    # Naudojamas jūsų originalus raginimas
    prompt = f"""
    Jūs esate dirbtinio intelekto asistentas, specializuojasi sąskaitų faktūrų duomenų ištraukime.
    Išanalizuokite šį sąskaitos faktūros tekstą ir ištraukite visą struktūrizuotą informaciją.

    Atsakymą pateikite TIK JSON formatu, be jokių papildomų paaiškinimų ar teksto.
    Niekada neįtraukite papildomų žodžių, frazių ar Markdown formatavimo (pvz., ```json) prieš JSON pradžią ar po pabaigos.

    Štai JSON struktūra, kurią turite naudoti:
    {{
      "dokumento_tipas": "PVM sąskaita faktūra",
      "numeris": "Sąskaitos numeris",
      "data": "Sąskaitos data (YYYY-MM-DD formatu)",
      "pardavejas": {{
        "pavadinimas": "Pardavėjo pavadinimas",
        "imones_kodas": "Įmonės kodas",
        "pvm_kodas": "PVM kodas",
        "adresas": "Adresas",
        "bankas": "Bankas",
        "saskaitos_numeris": "Sąskaitos numeris"
      }},
      "gavejas": {{
        "pavadinimas": "Gavėjo pavadinimas",
        "imones_kodas": "Įmonės kodas",
        "pvm_kodas": "PVM kodas",
        "adresas": "Adresas"
      }},
      "prekes": [
        {{
          "pavadinimas": "Prekės pavadinimas",
          "vezimas": "Važtaraščio numeris (jei nurodytas)",
          "kiekis_t": "Kiekis tonomis (skaičius, naudokite tašką kaip dešimtainį skirtuką)",
          "vieneto_kaina_eur": "Vieneto kaina eurais (skaičius, naudokite tašką kaip dešimtainį skirtuką)",
          "viso_eur": "Bendra kaina eurais (skaičius, naudokite tašką kaip dešimtainį skirtuką)"
        }}
      ],
      "sumos": {{
        "viso_be_pvm_eur": "Bendra suma be PVM (skaičius, naudokite tašką kaip dešimtainį skirtuką)",
        "pvm_suma_eur": "PVM suma (skaičius, naudokite tašką kaip dešimtainį skirtuką)",
        "viso_su_pvm_eur": "Bendra suma su PVM (skaičius, naudokite tašką kaip dešimtainį skirtuką)"
      }},
      "apmoketi_iki": "Apmokėjimo terminas (YYYY-MM-DD formatu)"
    }}

    Jei skaitinės reikšmės nerandamos, naudokite '0'. Jei tekstiniai laukai nerandami, palikite juos tuščius "".
    Įsitikinkite, kad grąžinate tik JSON kodą.
    NEĮTRAUKITE JOKIŲ PAPILDOMŲ KOMENTARŲ AR TEKSTO UŽ JSON STRUKTŪROS RIBŲ.

    Sąskaitos faktūros tekstas:
    ```
    {pdf_text}
    ```
    """
    return prompt


def get_contract_prompt(pdf_text):
    """
    Grąžina raginimą sutarties duomenų ištraukimui.
    """
    # Nauja JSON struktūra sutartims
    prompt = f"""
    Jūs esate dirbtinio intelekto asistentas, specializuojasi sutarčių duomenų ištraukime.
    Išanalizuokite šį sutarties tekstą ir ištraukite pagrindinius parametrus.

    Atsakymą pateikite TIK JSON formatu, be jokių papildomų paaiškinimų ar teksto.
    Niekada neįtraukite papildomų žodžių, frazių ar Markdown formatavimo (pvz., ```json) prieš JSON pradžią ar po pabaigos.

    Štai JSON struktūra, kurią turite naudoti:
    {{
      "dokumento_tipas": "Sutartis",
      "numeris": "Sutarties numeris (jei nurodytas)",
      "sudarymo_data": "Sutarties sudarymo data (YYYY-MM-DD formatu)",
      "sutarties_tipas": "Pirkimo-pardavimo, Nuomos, Paslaugų teikimo ar pan.",
      "salis_a": {{
        "pavadinimas": "Šalies A (Pardavėjo/Nuomotojo/Teikėjo) pavadinimas",
        "imones_kodas": "Įmonės kodas",
        "adresas": "Adresas"
      }},
      "salis_b": {{
        "pavadinimas": "Šalies B (Pirkėjo/Nuomininko/Gavėjo) pavadinimas",
        "imones_kodas": "Įmonės kodas",
        "adresas": "Adresas"
      }},
      "galiojimo_terminas": "Sutarties galiojimo terminas (pvz., 1 metai, Iki 2025-12-31, Neterminuota)",
      "bendra_suma_eur": "Bendra sutarties vertė eurais (skaičius, naudokite tašką kaip dešimtainį skirtuką. Jei nenaudojama, naudokite '0')",
      "mokestis_uz_paslaugas": "Mokestis už paslaugas/prekes (detalesnis aprašymas, pvz., '1200 EUR per mėnesį', '1.5 EUR už vienetą')"
    }}

    Jei skaitinės reikšmės nerandamos, naudokite '0'. Jei tekstiniai laukai nerandami, palikite juos tuščius "".
    Įsitikinkite, kad grąžinate tik JSON kodą.
    NEĮTRAUKITE JOKIŲ PAPILDOMŲ KOMENTARŲ AR TEKSTO UŽ JSON STRUKTŪROS RIBŲ.

    Sutarties tekstas:
    ```
    {pdf_text}
    ```
    """
    return prompt


def classify_document(pdf_text_sample):
    """
    Naudoja AI, kad klasifikuotų dokumento tipą (invoice arba contract).
    """
    classification_prompt = f"""
    Išanalizuokite šio dokumento tekstą ir nustatykite jo tipą.
    Jums reikia pasirinkti TIK iš šių dviejų variantų: 'invoice' (sąskaita faktūra) ARBA 'contract' (sutartis).

    Atsakymą pateikite TIK vienu žodžiu be jokių papildomų paaiškinimų, kabučių ar ženklų.
    Jei nerandate aiškaus tipo, grąžinkite 'unknown'.

    Dokumento tekstas:
    ```
    {pdf_text_sample[:1000]}
    ```
    """
    try:
        response = AI_MODEL.generate_content(classification_prompt)
        # Išvalome ir grąžiname atsakymą mažosiomis raidėmis
        classification = response.text.strip().lower()

        if classification in ["invoice", "contract"]:
            return classification
        else:
            print(f"Įspėjimas: Klasifikatorius grąžino nežinomą tipą: '{classification}'")
            return "unknown"

    except Exception as e:
        print(f"Klaida klasifikuojant dokumentą: {e}")
        return "unknown"


def process_pdf_with_ai(pdf_text, doc_type):
    """
    Siunčia PDF tekstą į Gemini AI ir prašo grąžinti JSON formatu,
    naudojant atitinkamą raginimą.
    """
    if doc_type == "invoice":
        prompt = get_invoice_prompt(pdf_text)
        print("  -> Naudojamas SĄSKAITOS FAKTŪROS raginimas.")
    elif doc_type == "contract":
        prompt = get_contract_prompt(pdf_text)
        print("  -> Naudojamas SUTARTIES raginimas.")
    else:
        print(f"Klaida: Nepalaikomas dokumento tipas: {doc_type}")
        return None

    response = None

    try:
        response = AI_MODEL.generate_content(prompt)

        # Rankinis JSON valymas (pašaliname '```json' ir '```', naudojame regex)
        # Tai yra tvirčiausias būdas išgauti JSON, net jei modelis prideda žymes.
        match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)

        if match:
            json_string = match.group(0)
        else:
            # Atsarginis valymas
            json_string = response.text.strip().lstrip('`json').strip('`')

        return json.loads(json_string)

    except json.JSONDecodeError as e:
        print(f"❌ Klaida dekoduojant JSON atsakymą: {e}")
        if response:
            print(f"Modelio grąžintas tekstas (pradžia):\n{response.text[:500]}...")
        return None
    except Exception as e:
        print(f"Klaida bendraujant su Gemini AI arba apdorojant atsakymą: {e}")
        return None


def process_folder(pdf_input_folder):
    """
    Pagrindinė funkcija, apdorojanti visus PDF failus aplanke,
    automatiškai klasifikuojanti ir išsauganti atitinkamuose aplankuose.
    """
    if not os.path.exists(pdf_input_folder):
        print(f"Informacija: Aplankas '{pdf_input_folder}' nerastas. Praleidžiama.")
        return

    pdf_files = [f for f in os.listdir(pdf_input_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Aplanke '{pdf_input_folder}' nerasta jokių PDF failų.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_input_folder, pdf_file)
        print(f"\n--- Apdorojamas failas: {pdf_file}...")

        # 1. Paimame tik dalį teksto klasifikavimui (greičiau)
        pdf_sample_text = extract_text_from_pdf(pdf_path)
        if not pdf_sample_text:
            print(f"Tekstas iš '{pdf_file}' neišgautas. Praleidžiama.")
            continue

        # 2. Klasifikuojame dokumentą
        doc_type = classify_document(pdf_sample_text)
        print(f"  -> Dokumento tipas nustatytas kaip: **{doc_type.upper()}**")

        if doc_type == "unknown":
            print(f"❌ Nepavyko nustatyti dokumento tipo: {pdf_file}. Jis nebuvo apdorotas.")
            continue

        # 3. Ištraukiame visą tekstą (jei reikia išsamiai analizei)
        full_pdf_text = extract_full_text_from_pdf(pdf_path)

        # 4. Apdorojame su AI, naudodami atitinkamą raginimą
        doc_json_data = process_pdf_with_ai(full_pdf_text, doc_type)

        if doc_json_data:
            json_file_name = pdf_file.replace('.pdf', '.json')

            # Pasirenkame išvesties aplanką pagal tipą
            if doc_type == "invoice":
                json_path = os.path.join(JSON_FOLDER_INVOICES, json_file_name)
            else:  # contract
                json_path = os.path.join(JSON_FOLDER_CONTRACTS, json_file_name)

            with open(json_path, 'w', encoding='utf-8') as f:
                # Naudojame 'ensure_ascii=False' kad išsaugotume lietuviškas raides
                json.dump(doc_json_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Sėkmingai sugeneruotas JSON failas: {json_file_name} į '{doc_type}' aplanką.")

            os.remove(pdf_path)
            print(f"🗑️ Originalus PDF failas '{pdf_file}' pašalintas.")
        else:
            print(f"❌ Nepavyko išgauti struktūrizuotų duomenų iš: {pdf_file}. Jis nebuvo pašalintas.")


def main():
    print("\n--- Pradedamas automatizuotas DOKUMENTŲ apdorojimas (SF/Sutartis) ---")
    # Visi failai dabar apdorojami iš vieno aplanko
    process_folder(PDF_FOLDER_DOCUMENTS)

    print("\n\n--- Visų dokumentų konvertavimas baigtas. ---")


if __name__ == "__main__":
    main()