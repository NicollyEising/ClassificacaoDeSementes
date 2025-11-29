import datetime
from flask import Flask, request, jsonify
from bancoDeDados import Database
from werkzeug.utils import secure_filename
import os
from psycopg2.extras import RealDictCursor
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # configura CORS para seu frontend

UPLOAD_FOLDER = "/tmp/bucket_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sessão simples em memória
usuarios_logados = {}

def get_db():
    return Database(dbname="sementesdb", user="postgres", password="123")

# ---------------- USUÁRIOS ----------------
@app.route("/cadastrar", methods=["POST"])
def cadastrar():
    dados = request.get_json()
    email = dados.get("email")
    senha = dados.get("senha")

    if not email or not senha:
        return jsonify({"erro": "Email e senha são obrigatórios."}), 400

    db = get_db()
    try:
        user_id = db.inserir_usuario(email, senha)
        return jsonify({"mensagem": "Usuário cadastrado com sucesso.", "id": user_id}), 201
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        db.fechar()


@app.route("/login", methods=["POST"])
def login():
    dados = request.get_json()
    email = dados.get("email")
    senha = dados.get("senha")

    if not email or not senha:
        return jsonify({"erro": "Email e senha são obrigatórios."}), 400

    db = get_db()
    try:
        user_id = db.autenticar_usuario(email, senha)
    finally:
        db.fechar()

    if user_id:
        usuarios_logados[user_id] = True
        return jsonify({"mensagem": "Login realizado com sucesso.", "id": user_id}), 200
    else:
        return jsonify({"erro": "Credenciais inválidas."}), 401


@app.route("/status/<int:user_id>", methods=["GET"])
def status(user_id):
    logado = usuarios_logados.get(user_id, False)
    return jsonify({"usuario_id": user_id, "logado": logado})


# ---------------- RESULTADOS ----------------
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import qrcode
import io
from datetime import datetime


@app.route("/resultados", methods=["POST"])
def inserir_resultado():
    """
    Endpoint para inserção de resultados de classificação de sementes.

    Recebe dados do formulário:
    - usuario_id: ID do usuário
    - classe_prevista: nome da classe prevista
    - probabilidade: probabilidade associada à classe
    - cidade, temperatura, condicao, chance_chuva: dados climáticos
    - imagem: arquivo de imagem da semente

    Gera QR code e URL detalhada, insere no banco e retorna ID, QR e URL.
    """
    usuario_id = request.form.get("usuario_id")
    classe = request.form.get("classe_prevista")
    probabilidade = request.form.get("probabilidade", type=float)
    cidade = request.form.get("cidade")
    temperatura = request.form.get("temperatura", type=float)
    condicao = request.form.get("condicao")
    chance_chuva = request.form.get("chance_chuva", type=float)
    arquivo = request.files.get("imagem")
    img_bytes = arquivo.read() if arquivo else None
    nome_arquivo = secure_filename(arquivo.filename) if arquivo else None

    # gera URL e QR code primeiro
    # usando ID temporário fictício para URL, será substituído após INSERT
    semente_id_temp = "TEMP"  # apenas para gerar QR
    url_pagina_temp = f"http://127.0.0.1:5500/frontend/item.html?id={semente_id_temp}"
    qr = qrcode.QRCode(version=1, box_size=8, border=4)
    qr.add_data(url_pagina_temp)
    qr.make(fit=True)
    buffer = io.BytesIO()
    qr.make_image(fill_color="black", back_color="white").save(buffer, format="PNG")
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()

    db = get_db()
    try:
        # INSERE já com URL e QR
        semente_id = db.inserir_resultado(
            usuario_id=usuario_id,
            img_bytes=img_bytes,
            classe_prevista=classe,
            probabilidade=probabilidade,
            cidade=cidade,
            temperatura=temperatura,
            condicao=condicao,
            chance_chuva=chance_chuva,
            nome_arquivo=nome_arquivo,
            data_hora=datetime.now(),
            url_detalhes=None,       # será atualizado imediatamente após pegar id real
            qrcode_base64=None
        )

        # agora que temos o id real, geramos URL correta
        url_pagina = f"http://127.0.0.1:5500/frontend/item.html?id={semente_id}"
        qr = qrcode.QRCode(version=1, box_size=8, border=4)
        qr.add_data(url_pagina)
        qr.make(fit=True)
        buffer = io.BytesIO()
        qr.make_image(fill_color="black", back_color="white").save(buffer, format="PNG")
        qr_base64 = base64.b64encode(buffer.getvalue()).decode()

        # atualizar registro com URL real e QR
        db.atualizar_resultado_qr(semente_id, qr_base64, url_pagina)

        return jsonify({
            "mensagem": "Resultado inserido com sucesso.",
            "semente_id": semente_id,
            "qrcode_base64": qr_base64,
            "url_detalhes": url_pagina
        }), 201

    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        db.fechar()



@app.route("/resultados", methods=["GET"])
def listar_resultados():
    """
    Endpoint para listar resultados de sementes.

    Parâmetros:
    - limite (query param, opcional): número máximo de registros a retornar (default=10)

    Retorna lista de resultados com imagens codificadas em base64.
    """
    limite = request.args.get("limite", default=10, type=int)
    db = get_db()
    try:
        resultados = db.listar_resultados(limite)
        # converte bytes da imagem para base64 para JSON, se houver
        for r in resultados:
            if r["imagem"]:
                import base64
                r["imagem"] = base64.b64encode(r["imagem"]).decode()
        return jsonify(resultados)
    finally:
        db.fechar()

@app.route("/resultados/<int:usuario_id>", methods=["GET"])
def resultados_por_usuario(usuario_id):
    """
    Endpoint para listar todos os resultados de um usuário específico.

    Parâmetros:
    - usuario_id (path param): ID do usuário cujos resultados serão listados

    Retorna uma lista de resultados com imagens codificadas em base64 e datas no formato ISO 8601.
    """
    db = get_db()
    try:
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.imagem, r.nome_arquivo, r.classe_prevista, r.probabilidade,
                       r.data_hora, r.cidade, r.temperatura, r.condicao, r.chance_chuva, r.url_detalhes, r.qrcode_base64,
                       u.email AS usuario_email
                FROM resultados r
                JOIN usuarios u ON r.usuario_id = u.id
                WHERE r.usuario_id = %s
                ORDER BY r.id DESC;
            """, (usuario_id,))
            resultados = cur.fetchall()

        if not resultados:
            return jsonify({"erro": "Nenhum resultado encontrado para este usuário"}), 404

        # Converte imagens para base64 e datas para string ISO
        for r in resultados:
            if r["imagem"]:
                r["imagem"] = base64.b64encode(r["imagem"]).decode()
            if r["data_hora"]:
                r["data_hora"] = r["data_hora"].isoformat()  # formata para string ISO 8601

        return jsonify(resultados)
    finally:
        db.fechar()

@app.route("/resultado/<int:resultado_id>", methods=["GET"])
def resultado_por_id(resultado_id):
    """
    Endpoint para obter detalhes de um resultado específico pelo seu ID.

    Parâmetros:
    - resultado_id (path param): ID do resultado a ser recuperado

    Retorna o registro completo com:
    - imagem em base64
    - informações de clima
    - dados do usuário
    - QR code e URL de detalhes
    - datas formatadas no padrão ISO 8601
    """
    db = get_db()
    try:
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT r.id, r.imagem, r.nome_arquivo, r.classe_prevista, r.probabilidade,
                       r.data_hora, r.cidade, r.temperatura, r.condicao, r.chance_chuva,
                       r.qrcode_base64, r.url_detalhes,
                       u.email AS usuario_email
                FROM resultados r
                JOIN usuarios u ON r.usuario_id = u.id
                WHERE r.id = %s;
            """, (resultado_id,))
            resultado = cur.fetchone()

        if not resultado:
            return jsonify({"erro": "Resultado não encontrado"}), 404

        # Converte imagem binária para base64 e formata a data
        if resultado["imagem"]:
            resultado["imagem"] = base64.b64encode(resultado["imagem"]).decode()
        if resultado["data_hora"]:
            resultado["data_hora"] = resultado["data_hora"].isoformat()

        return jsonify(resultado), 200

    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        db.fechar()


# ---------------- CHECKPOINTS VAE ----------------
@app.route("/checkpoints/<categoria>", methods=["POST"])
def salvar_checkpoint(categoria):
    """
    Endpoint para salvar checkpoints de modelos VAE por categoria.

    Parâmetros:
    - categoria (path param): categoria do VAE (ex: tipo de semente)
    - checkpoint (form file): arquivo de checkpoint a ser salvo
    - best_model (form file): arquivo do melhor modelo a ser salvo

    Retorna:
    - JSON com mensagem de sucesso e IDs gerados no banco de dados
    """
    ckpt_file = request.files.get("checkpoint")
    best_file = request.files.get("best_model")
    if not ckpt_file or not best_file:
        return jsonify({"erro": "Arquivos checkpoint e best_model são obrigatórios"}), 400

    ckpt_path = os.path.join(UPLOAD_FOLDER, secure_filename(ckpt_file.filename))
    best_path = os.path.join(UPLOAD_FOLDER, secure_filename(best_file.filename))
    ckpt_file.save(ckpt_path)
    best_file.save(best_path)

    db = get_db()
    try:
        oids = db.salvar_checkpoint_vae(categoria, ckpt_path, best_path)
        return jsonify({"mensagem": "Checkpoint salvo", **oids}), 201
    finally:
        db.fechar()


@app.route("/checkpoints/<categoria>", methods=["GET"])
def recuperar_checkpoint(categoria):
    ckpt_out = os.path.join(UPLOAD_FOLDER, f"{categoria}_checkpoint.bin")
    best_out = os.path.join(UPLOAD_FOLDER, f"{categoria}_best_model.bin")
    db = get_db()
    try:
        db.recuperar_checkpoint_vae_lo(categoria, ckpt_out, best_out)
        return jsonify({"checkpoint_path": ckpt_out, "best_model_path": best_out})
    finally:
        db.fechar()


@app.route("/checkpoints/<categoria>", methods=["DELETE"])
def remover_checkpoint(categoria):
    db = get_db()
    try:
        sucesso = db.remover_checkpoint_vae(categoria)
        if sucesso:
            return jsonify({"mensagem": "Checkpoint removido"}), 200
        else:
            return jsonify({"erro": "Categoria não encontrada"}), 404
    finally:
        db.fechar()




if __name__ == "__main__":
    app.run(debug=True)