import sys
import os
import psycopg2
from psycopg2 import sql

DB_CONFIG = {
    "dbname": "sementesdb",
    "user": "postgres",
    "password": "123",
    "host": "localhost",
    "port": 5432
}

def recuperar_imagem_por_id(conn, rec_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, nome_arquivo, octet_length(imagem) AS tamanho, imagem "
            "FROM resultados WHERE id = %s", (rec_id,)
        )
        return cur.fetchone()

def recuperar_imagem_mais_recente(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, nome_arquivo, octet_length(imagem) AS tamanho, imagem "
            "FROM resultados WHERE imagem IS NOT NULL ORDER BY data_hora DESC LIMIT 1"
        )
        return cur.fetchone()

def salvar_e_abrir(row):
    if not row:
        print("Nenhum registro encontrado.")
        return

    rec_id, nome_arquivo, tamanho, imagem_blob = row
    if not imagem_blob:
        print(f"Registro {rec_id} não possui imagem (imagem_blob vazio).")
        return

    out_name = f"recuperada_{rec_id}.png"
    with open(out_name, "wb") as f:
        f.write(bytes(imagem_blob))
    print(f"Imagem salva em: {os.path.abspath(out_name)}  (bytes: {tamanho})")

    # Abrir com o visualizador padrão (Windows/macOS/Linux)
    try:
        if os.name == "nt":
            os.startfile(out_name)           # Windows
        else:
            # macOS / Linux
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            os.spawnlp(os.P_NOWAIT, opener, opener, out_name)
    except Exception as e:
        print("Falha ao abrir automaticamente a imagem:", e)
        print("Abra manualmente o arquivo:", os.path.abspath(out_name))

def main():
    rec_id = None
    if len(sys.argv) >= 2:
        try:
            rec_id = int(sys.argv[1])
        except ValueError:
            print("Argumento inválido. Use: python mostrar_imagem_db.py [id]")
            return

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        if rec_id is not None:
            row = recuperar_imagem_por_id(conn, rec_id)
        else:
            row = recuperar_imagem_mais_recente(conn)

        salvar_e_abrir(row)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
