# bancoDeDados.py
import base64
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import hashlib
import os

class Database:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        self.conn = psycopg2.connect(
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        self._create_tables()

    def _create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    senha_hash TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS resultados (
                    id SERIAL PRIMARY KEY,
                    usuario_id INT REFERENCES usuarios(id) ON DELETE CASCADE,
                    imagem BYTEA,
                    nome_arquivo TEXT,
                    classe_prevista TEXT,
                    probabilidade REAL,
                    data_hora TIMESTAMP,
                    cidade TEXT,
                    temperatura REAL,
                    condicao TEXT,
                    chance_chuva REAL,
                    url_detalhes TEXT,
                    qrcode_base64 TEXT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS resultados_treinamento (
                    id SERIAL PRIMARY KEY,
                    data_hora TIMESTAMP NOT NULL,
                    epochs INT,
                    best_val_acc REAL,
                    f1_weighted REAL,
                    iou REAL,
                    class_report JSONB,
                    train_samples INT,
                    val_samples INT
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints_modelos_vae (
                    id SERIAL PRIMARY KEY,
                    categoria VARCHAR(100) UNIQUE,
                    checkpoint_oid BIGINT,
                    best_model_oid BIGINT,
                    data_criacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.conn.commit()

    # ---------------- USUÁRIOS ----------------
    def inserir_usuario(self, email, senha):
        senha_hash = hashlib.sha256(senha.encode()).hexdigest()
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM usuarios WHERE email = %s;", (email,))
            if cur.fetchone():
                raise ValueError("E-mail já cadastrado")
            cur.execute(
                "INSERT INTO usuarios (email, senha_hash) VALUES (%s, %s) RETURNING id;",
                (email, senha_hash)
            )
            user_id = cur.fetchone()[0]
        self.conn.commit()
        return user_id

    def autenticar_usuario(self, email, senha):
        senha_hash = hashlib.sha256(senha.encode('utf-8')).hexdigest()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM usuarios WHERE email = %s AND senha_hash = %s;",
                (email, senha_hash)
            )
            row = cur.fetchone()
        return row[0] if row else None

    # ---------------- RESULTADOS ----------------
    def inserir_resultado(
        self,
        usuario_id,
        img_bytes,
        classe_prevista,
        probabilidade,
        cidade,
        temperatura,
        condicao,
        chance_chuva,
        nome_arquivo,
        data_hora,
        url_detalhes,
        qrcode_base64
    ):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO resultados (
                        usuario_id,
                        imagem,
                        classe_prevista,
                        probabilidade,
                        cidade,
                        temperatura,
                        condicao,
                        chance_chuva,
                        nome_arquivo,
                        data_hora,
                        url_detalhes,
                        qrcode_base64
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        usuario_id,
                        img_bytes,
                        classe_prevista,
                        probabilidade,
                        cidade,
                        temperatura,
                        condicao,
                        chance_chuva,
                        nome_arquivo,
                        data_hora,
                        url_detalhes,
                        qrcode_base64
                    )
                )
                semente_id = cur.fetchone()[0]

            self.conn.commit()
            return semente_id

        except Exception as e:
            print(f"ERRO EM inserir_resultado: {e}")
            self.conn.rollback()
            raise


    def obter_resultado_por_id(self, id: int):
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM resultados WHERE id = %s", (id,))
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "classe_prevista": row["classe_prevista"],
                "probabilidade": row["probabilidade"],
                "imagem_anotada_base64": base64.b64encode(row["img_bytes"]).decode("utf-8") if row["img_bytes"] else None,
                "clima": {
                    "cidade": row["cidade"],
                    "temperatura": row["temperatura"],
                    "condicao": row["condicao"],
                    "chance_chuva": row["chance_chuva"]
                },
                "recomendacoes": []  # ajuste conforme necessário
            }

    def obter_proximo_id(self) -> int:
        """
        Retorna o próximo ID que será inserido na tabela 'resultados'.
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT nextval(pg_get_serial_sequence('resultados', 'id'))")
            next_id = cur.fetchone()[0]
        return next_id

    def atualizar_resultado_qr(self, semente_id, qr_b64, url):
        """
        Atualiza qrcode_base64 e url_detalhes para o resultado indicado.
        Esta função levanta exceção se nenhuma linha for atualizada.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE resultados
                    SET qrcode_base64 = %s, url_detalhes = %s
                    WHERE id = %s
                    """,
                    (qr_b64, url, semente_id)
                )
                if cur.rowcount == 0:  # nenhuma linha atualizada -> id inexistente
                    raise ValueError(f"Nenhuma linha atualizada para id={semente_id}")
            # commit aqui garante persistência mesmo que chamador esqueça
            self.conn.commit()
        except Exception as e:
            # rollback para não deixar transação a meio
            try:
                self.conn.rollback()
            except Exception:
                pass
            # re-lança para o chamador ver o erro
            raise

    def listar_resultados(self, limite=10):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.imagem, r.nome_arquivo, r.classe_prevista, r.probabilidade,
                       r.data_hora, r.cidade, r.temperatura, r.condicao, r.chance_chuva,
                       r.url_detalhes, r.qrcode_base64, u.email AS usuario_email
                FROM resultados r
                JOIN usuarios u ON r.usuario_id = u.id
                ORDER BY r.data_hora DESC
                LIMIT %s;
            """, (limite,))
            return cur.fetchall()

    # ---------------- LARGE OBJECTS (checkpoints) ----------------
    def salvar_checkpoint_vae(self, categoria, checkpoint_path, best_model_path):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT checkpoint_oid, best_model_oid FROM checkpoints_modelos_vae WHERE categoria = %s FOR UPDATE", (categoria,))
                row = cur.fetchone()
                old_ckpt_oid, old_best_oid = (row or (None, None))

                lo_ckpt = self.conn.lobject(0, "wb")
                with open(checkpoint_path, "rb") as f:
                    while chunk := f.read(1024*1024):
                        lo_ckpt.write(chunk)
                new_ckpt_oid = lo_ckpt.oid
                lo_ckpt.close()

                lo_best = self.conn.lobject(0, "wb")
                with open(best_model_path, "rb") as f:
                    while chunk := f.read(1024*1024):
                        lo_best.write(chunk)
                new_best_oid = lo_best.oid
                lo_best.close()

                cur.execute("""
                    INSERT INTO checkpoints_modelos_vae (categoria, checkpoint_oid, best_model_oid, data_criacao)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (categoria) DO UPDATE
                    SET checkpoint_oid = EXCLUDED.checkpoint_oid,
                        best_model_oid = EXCLUDED.best_model_oid,
                        data_criacao = EXCLUDED.data_criacao;
                """, (categoria, new_ckpt_oid, new_best_oid))

                if old_ckpt_oid:
                    cur.execute("SELECT lo_unlink(%s)", (old_ckpt_oid,))
                if old_best_oid:
                    cur.execute("SELECT lo_unlink(%s)", (old_best_oid,))
            self.conn.commit()
            return {"checkpoint_oid": new_ckpt_oid, "best_model_oid": new_best_oid}
        except Exception:
            if getattr(self.conn, "closed", 1) == 0:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            raise

    def recuperar_checkpoint_vae_lo(self, categoria, out_checkpoint_path, out_best_model_path):
        with self.conn.cursor() as cur:
            cur.execute("SELECT checkpoint_oid, best_model_oid FROM checkpoints_modelos_vae WHERE categoria = %s", (categoria,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Nenhum checkpoint encontrado para a categoria '{categoria}'")
            ckpt_oid, best_oid = row

            if ckpt_oid:
                lo_ckpt = self.conn.lobject(ckpt_oid, "rb")
                with open(out_checkpoint_path, "wb") as f:
                    while chunk := lo_ckpt.read(1024*1024):
                        f.write(chunk)
                lo_ckpt.close()

            if best_oid:
                lo_best = self.conn.lobject(best_oid, "rb")
                with open(out_best_model_path, "wb") as f:
                    while chunk := lo_best.read(1024*1024):
                        f.write(chunk)
                lo_best.close()

    def remover_checkpoint_vae(self, categoria):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT checkpoint_oid, best_model_oid FROM checkpoints_modelos_vae WHERE categoria = %s FOR UPDATE", (categoria,))
                row = cur.fetchone()
                if not row:
                    return False
                ckpt_oid, best_oid = row
                if ckpt_oid:
                    cur.execute("SELECT lo_unlink(%s)", (ckpt_oid,))
                if best_oid:
                    cur.execute("SELECT lo_unlink(%s)", (best_oid,))
                cur.execute("DELETE FROM checkpoints_modelos_vae WHERE categoria = %s", (categoria,))
            self.conn.commit()
            return True
        except Exception:
            if getattr(self.conn, "closed", 1) == 0:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
            raise

    # ---------------- CONEXÃO ----------------
    def fechar(self):
        if getattr(self.conn, "closed", 1) == 0:
            self.conn.close()



