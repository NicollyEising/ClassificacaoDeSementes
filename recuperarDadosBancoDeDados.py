import json
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


def recuperar_treinamento_por_id(conn, rec_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, data_hora, epochs, best_val_acc, f1_weighted, iou, class_report, train_samples, val_samples
            FROM resultados_treinamento
            WHERE id = %s
            """,
            (rec_id,)
        )
        return cur.fetchone()

def recuperar_treinamento_mais_recente(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, data_hora, epochs, best_val_acc, f1_weighted, iou, class_report, train_samples, val_samples
            FROM resultados_treinamento
            ORDER BY data_hora DESC
            LIMIT 1
            """
        )
        return cur.fetchone()

def mostrar_resultado_treinamento(row):
    if not row:
        print("Nenhum registro encontrado.")
        return

    (rec_id, data_hora, epochs, best_val_acc, f1_weighted, iou,
     class_report, train_samples, val_samples) = row

    print(f"\n=== Resultado de Treinamento ID {rec_id} ===")
    print(f"Data/Hora: {data_hora}")
    print(f"Épocas: {epochs}")
    print(f"Acurácia Validação: {best_val_acc:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Amostras Treino: {train_samples}")
    print(f"Amostras Validação: {val_samples}")

    try:
        report = json.loads(class_report)
        print("\n--- Relatório por Classe ---")
        for classe, metricas in report.items():
            if isinstance(metricas, dict):
                rec = metricas.get("recall", None)
                prec = metricas.get("precision", None)
                f1 = metricas.get("f1-score", None)
                if rec is not None and prec is not None and f1 is not None:
                    print(f"{classe:25s} | Recall: {rec:.3f} | Precision: {prec:.3f} | F1: {f1:.3f}")
    except Exception as e:
        print("Não foi possível decodificar o class_report:", e)



def recuperar_checkpoint_vae(self, categoria, salvar_checkpoint="tmp_checkpoint.pth", salvar_best="tmp_best.pth"):
    self.cur.execute("""
        SELECT checkpoint, best_model FROM checkpoints_modelos_vae
        WHERE categoria = %s
    """, (categoria,))
    row = self.cur.fetchone()
    if row:
        with open(salvar_checkpoint, "wb") as f_ckpt:
            f_ckpt.write(row[0])
        with open(salvar_best, "wb") as f_best:
            f_best.write(row[1])
        return salvar_checkpoint, salvar_best
    return None, None


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

    rec_id = None
    if len(sys.argv) >= 2:
        try:
            rec_id = int(sys.argv[1])
        except ValueError:
            print("Argumento inválido. Use: python mostrar_treinamento.py [id]")
            return

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        if rec_id is not None:
            row = recuperar_treinamento_por_id(conn, rec_id)
        else:
            row = recuperar_treinamento_mais_recente(conn)

        mostrar_resultado_treinamento(row)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
