"""
Hamming (12,8) – koder + dekoder + diagnostyka w konsoli.
Konwencja jak na slajdzie:
  - pozycje 1..12 (NIE indeksy!), pozycja 12 = MSB (lewa strona tabeli)
  - bity kontrolne na pozycjach 1, 2, 4, 8  (C1, C2, C4, C8)
  - bity danych D1..D8 na pozycjach 3, 5, 6, 7, 9, 10, 11, 12
  - wiadomość '00111001' czytamy jako D8 D7 ... D1 (D1 = ostatni znak)

Uruchomienie:
  python hamming_encoder.py                      -> testy + demo ze slajdu (błąd na poz. 6)
  python hamming_encoder.py 10101010             -> własna wiadomość, bez błędów
  python hamming_encoder.py 10101010 -e 5        -> błąd na pozycji 5
  python hamming_encoder.py 00111001 -e 3 5      -> dwa błędy (pokazuje, czemu Hamming wtedy zawodzi)
  python hamming_encoder.py 00111001 -r 1        -> 1 losowy błąd
"""
import argparse
import random
import numpy as np

N, K = 12, 8
CHECK_POS = [1, 2, 4, 8]
DATA_POS = [p for p in range(1, N + 1) if p not in CHECK_POS]   # [3,5,6,7,9,10,11,12]


def str_to_bits(s: str) -> np.ndarray:
    """'D8...D1' -> tablica bool, gdzie [0] = D1."""
    return np.array([c == '1' for c in reversed(s)], dtype=bool)


def block_to_str(block: np.ndarray) -> str:
    """Blok (długość 13, [0] nieużywane) -> napis od pozycji 12 do 1, jak w tabeli."""
    return ''.join('1' if block[p] else '0' for p in range(N, 0, -1))


def xor_of_positions(block: np.ndarray) -> int:
    """XOR (jako liczba całkowita) numerów wszystkich pozycji, na których jest '1'.
    To jest dokładnie "XOR w pionie" z tabel (b) i (d)."""
    acc = 0
    for i in range(1, N + 1):          # POPRAWKA: było range(1, block+1) – block to tablica, nie liczba
        if block[i]:
            acc ^= i
    return acc


def encode(data: str) -> np.ndarray:
    """8 bitów danych -> 13-elementowy blok (pozycje 1..12)."""
    d = str_to_bits(data)
    block = np.zeros(N + 1, dtype=bool)
    for bit, pos in zip(d, DATA_POS):
        block[pos] = bit

    checks = xor_of_positions(block)   # POPRAWKA: wynik był liczony, ale nigdzie nie trafiał
    for i, pos in enumerate(CHECK_POS):    # bit i wyniku -> pozycja 2**i
        block[pos] = (checks >> i) & 1
    return block


def classify_syndrome(syn: int) -> str:
    """Słowny opis tego, co mówi syndrom."""
    if syn == 0:
        return "brak błędu"
    if syn > N:
        return f"syndrom {syn} wskazuje nieistniejącą pozycję -> co najmniej 2 błędy, NIE da się poprawić"
    if syn in CHECK_POS:
        return f"błąd w bicie kontrolnym C{syn} (poz. {syn}) – dane są OK"
    return f"błąd w bicie danych D{DATA_POS.index(syn) + 1} (poz. {syn}) – poprawiam"


def extract_data(block: np.ndarray) -> str:
    """Bity z DATA_POS -> napis D8..D1."""
    return ''.join('1' if block[p] else '0' for p in reversed(DATA_POS))


def decode(block: np.ndarray):
    """Zwraca (dane_8bit_str, syndrom, poprawiony_blok)."""
    block = block.copy()

    # 1. syndrom = XOR numerów pozycji z jedynką w ODEBRANYM bloku (razem z bitami kontrolnymi).
    #    Dla poprawnego bloku = 0, bo koder dobrał C1..C8 właśnie tak, żeby się wyzerowało.
    syn = xor_of_positions(block)

    # 2. korekcja – tylko jeśli syndrom wskazuje istniejącą pozycję
    if 1 <= syn <= N:
        block[syn] = not block[syn]
    # syn > N (13..15): wiemy, że są >= 2 błędy, ale nie wiemy gdzie – nic nie ruszamy

    # 3. wyciągnięcie danych
    return extract_data(block), syn, block


# ------------------------- diagnostyka -------------------------
def label(p: int) -> str:
    return f"C{p}" if p in CHECK_POS else f"D{DATA_POS.index(p) + 1}"


def print_block(title: str, block: np.ndarray, mark=()):
    """Tabela jak na slajdzie: pozycja / numer binarny / D-C / wartość. `mark` – pozycje do podświetlenia."""
    pos = range(N, 0, -1)
    print(f"  {title}")
    print("    Pozycja : " + " ".join(f"{p:>4}" for p in pos))
    print("    Binarnie: " + " ".join(f"{p:04b}" for p in pos))
    print("    Bit     : " + " ".join(f"{label(p):>4}" for p in pos))
    print("    Wartość : " + " ".join(f"{int(block[p]):>4}" for p in pos))
    if mark:
        print("              " + " ".join("   ^" if p in mark else "    " for p in pos))


def print_xor_table(title: str, block: np.ndarray, result_name: str):
    """Tabela (b)/(d): pozycje z jedynką i XOR w pionie."""
    ones = [p for p in range(N, 0, -1) if block[p]]
    print(f"  {title}")
    for p in ones:
        print(f"    {p:>2} ({label(p):>3})  {p:04b}")
    print(f"    {'-' * 17}")
    res = xor_of_positions(block)
    print(f"    {result_name:<9} {res:04b}  (= {res})")


def run_demo(msg: str, err_positions):
    line = "=" * 70
    print(line)
    print(f"WIADOMOŚĆ WEJŚCIOWA (D8..D1): {msg}")
    print(line)

    # --- koder ---
    data_only = np.zeros(N + 1, dtype=bool)
    for bit, pos in zip(str_to_bits(msg), DATA_POS):
        data_only[pos] = bit
    print("\n[KODER]")
    print_xor_table("XOR pozycji bitów danych = 1 (tabela b):", data_only, "C8C4C2C1")
    tx = encode(msg)
    print()
    print_block("Blok nadany:", tx)
    print(f"  Wyjście kodera: {block_to_str(tx)}")

    # --- kanał ---
    rx = tx.copy()
    for p in err_positions:
        rx[p] = not rx[p]
    print("\n[KANAŁ]")
    if err_positions:
        print(f"  Przekłamane pozycje: {sorted(err_positions)}")
        print_block("Blok odebrany:", rx, mark=err_positions)
    else:
        print("  Bez błędów")
    print(f"  Odebrano:       {block_to_str(rx)}")

    # --- dekoder ---
    print("\n[DEKODER]")
    print_xor_table("XOR pozycji wszystkich bitów = 1 (tabela d):", rx, "syndrom")
    out, syn, fixed = decode(rx)
    print(f"  Diagnoza: {classify_syndrome(syn)}")
    if 1 <= syn <= N:
        print_block("Blok po korekcji:", fixed, mark=[syn])
    print(f"  Wiadomość wyjściowa: {out}")

    # --- werdykt ---
    print()
    if out == msg:
        print(f"WYNIK: OK – odebrano dokładnie to, co nadano ({out})")
    else:
        diff = [f"D{8 - i}" for i, (a, b) in enumerate(zip(msg, out)) if a != b]
        print(f"WYNIK: BŁĄD – nadano {msg}, odebrano {out} (różne bity: {', '.join(diff)})")
        if len(err_positions) >= 2 and 1 <= syn <= N:
            print(f"  Uwaga: syndrom {syn} = XOR pozycji błędów {sorted(err_positions)}, więc dekoder")
            print(f"  'poprawił' niewinny bit {syn} i dołożył kolejny błąd – kod koryguje tylko 1 błąd.")
    print(line)


# ---------------------------- testy ----------------------------
def run_tests():
    b = encode('00111001')
    assert block_to_str(b) == '001101001111', block_to_str(b)

    for data, expected in [('10101010', '101001011000'),
                           ('11111111', '111101110111'),
                           ('00000001', '000000000111')]:
        assert block_to_str(encode(data)) == expected, data

    rx = encode('00111001'); rx[6] ^= True
    data, syn, _ = decode(rx)
    assert syn == 6 and data == '00111001'

    for v in range(256):
        msg = format(v, '08b')
        assert decode(encode(msg)) [1] == 0, msg           # bez błędu -> syndrom 0
        for p in range(1, N + 1):
            rx = encode(msg); rx[p] ^= True
            assert decode(rx)[0] == msg, (msg, p)

    print('Wszystkie testy OK\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Hamming (12,8) – demo")
    ap.add_argument("msg", nargs="?", default="00111001", help="8 bitów, D8..D1")
    ap.add_argument("-e", "--errors", type=int, nargs="*", default=None, help="pozycje błędów (1..12)")
    ap.add_argument("-r", "--random", type=int, default=0, help="liczba losowych błędów")
    args = ap.parse_args()

    if len(args.msg) != K or set(args.msg) - {'0', '1'}:
        ap.error("wiadomość musi mieć dokładnie 8 znaków 0/1")

    run_tests()

    if args.random:
        errs = random.sample(range(1, N + 1), args.random)
    elif args.errors is not None:
        errs = args.errors
    else:
        errs = [6] if args.msg == "00111001" else []   # domyślnie przykład ze slajdu
    if any(not 1 <= p <= N for p in errs):
        ap.error("pozycje błędów muszą być z zakresu 1..12")

    run_demo(args.msg, errs)