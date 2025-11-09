import subprocess
import sys
from typing import List


def paleisti_ir_transliuoti(failo_vardas: str) -> bool:
    """
    Paleidžia nurodytą Python failą ir transliuoja išvestį realiuoju laiku.
    Grąžina True, jei vykdymas sėkmingas (grįžimo kodas 0), False priešingu atveju.
    """
    print("=" * 60)
    print(f"[{failo_vardas.upper()}] PALEIDIMAS PRADEDAMAS...")
    print("=" * 60)

    try:
        # Popen paleidžia procesą ir nukreipia jo stdout/stderr tiesiai į dabartinę konsolę (transliavimas)
        procesas = subprocess.Popen(
            [sys.executable, failo_vardas],
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=1
        )

        # Laukia, kol procesas baigs darbą
        grizimo_kodas = procesas.wait()

        if grizimo_kodas == 0:
            print(f"\n✅ [{failo_vardas.upper()}] SĖKMINGAI UŽBAIGTAS. Grįžimo kodas: 0")
            return True
        else:
            print(f"\n❌ [{failo_vardas.upper()}] UŽBAIGTAS SU KLAIDA. Grįžimo kodas: {grizimo_kodas}")
            return False

    except FileNotFoundError:
        print(f"\nKLAIDA: Python interpretatorius ({sys.executable}) arba failas {failo_vardas} nerastas.")
        return False
    except Exception as e:
        print(f"\nĮvyko netikėta klaida paleidžiant {failo_vardas}: {e}")
        return False


# Pagrindinė vykdymo funkcija
if __name__ == "__main__":

    # Apibrėžiame vykdytinų failų seką
    vykdomu_failu_sarasas: List[str] = [
        "ai_pdf_to_json.py",
        "main.py",
        "app_local.py"
    ]

    viskas_sekminga: bool = True

    for i, failo_vardas in enumerate(vykdomu_failu_sarasas):

        if not viskas_sekminga:
            print("\n" * 2)
            print("*" * 60)
            print(f"SEKA NUTRAUKTA: Ankstesnis žingsnis nepavyko. Failas {failo_vardas} nebus paleistas.")
            print("*" * 60)
            break

        # Paleidžiame dabartinį failą ir tikriname sėkmę
        sekme = paleisti_ir_transliuoti(failo_vardas)

        if not sekme:
            viskas_sekminga = False
            # Paliekame `for` ciklą, kad būtų galima atspausdinti nutraukimo žinutę

        # Tarp žingsnių pridedame tarpą
        if viskas_sekminga and i < len(vykdomu_failu_sarasas) - 1:
            print("\n" * 3)
            print("--- Sekantis žingsnis ---")
            print("\n" * 3)

    if viskas_sekminga:
        print("\n" * 3)
        print("**************************************************")
        print("🥳 VISA PROGRAMOS VYKDYMO SEKA SĖKMINGAI BAIGTA.")
        print("**************************************************")