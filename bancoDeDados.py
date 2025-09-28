import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from datetime import datetime


class Database:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.cur = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS resultados (
            id SERIAL PRIMARY KEY,
            imagem BYTEA,
            nome_arquivo TEXT,
            classe_prevista TEXT,
            probabilidade REAL,
            data_hora TIMESTAMP,
            cidade TEXT,
            temperatura REAL,
            condicao TEXT,
            chance_chuva REAL
        );""")
        self.conn.commit()

    def inserir_resultado(self, img_bytes, classe_prevista, probabilidade,
                          cidade=None, temperatura=None, condicao=None, chance_chuva=None, nome_arquivo=None):
        data_hora = datetime.now()
        img_db = psycopg2.Binary(img_bytes) if img_bytes is not None else None
        self.cur.execute("""
            INSERT INTO resultados (imagem, nome_arquivo, classe_prevista, probabilidade,
                                    data_hora, cidade, temperatura, condicao, chance_chuva)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (img_db, nome_arquivo, classe_prevista, probabilidade,
              data_hora, cidade, temperatura, condicao, chance_chuva))
        self.conn.commit()

    def listar_resultados(self, limite=10):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, imagem, nome_arquivo, classe_prevista, probabilidade, data_hora,
                       cidade, temperatura, condicao, chance_chuva
                FROM resultados
                ORDER BY data_hora DESC
                LIMIT %s
            """, (limite,))
            return cur.fetchall()  # retorna lista de dicts

    def fechar(self):
        self.cur.close()
        self.conn.close()


if __name__ == "__main__":
    db = Database(dbname="sementesdb", user="seu_usuario", password="sua_senha")

    # Exemplo de inserção
    db.inserir_resultado(
        imagem_path="saida_gradcam.png",
        classe_prevista="Intact soybean",
        probabilidade=0.987,
        cidade="São Paulo",
        temperatura=25.3,
        condicao="Ensolarado",
        chance_chuva=40.0
    )

    # Recuperar a última imagem do banco e salvar no disco
    res = db.listar_resultados(limite=1)
    if res:
        row = res[0]
        _id = row[0]
        imagem_blob = row[1]
        if imagem_blob:
            with open(f"dump_{_id}.png", "wb") as f:
                f.write(bytes(imagem_blob))
            print("gravado dump_", _id)
        else:
            print("registro sem imagem")