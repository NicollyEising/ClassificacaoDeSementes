from getpass import getpass
from bancoDeDados import Database  # supondo que o código anterior esteja salvo em database.py


def menu():
    print("\n=== Sistema de Login ===")
    print("1 - Cadastrar novo usuário")
    print("2 - Fazer login")
    print("3 - Sair")
    return input("Escolha uma opção: ")


def cadastrar(db):
    email = input("Digite o e-mail: ")
    senha = getpass("Digite a senha: ")  # oculta senha no terminal
    try:
        user_id = db.inserir_usuario(email, senha)
        print(f"Usuário cadastrado com id {user_id}")
    except Exception as e:
        print("Erro ao cadastrar usuário:", e)


usuario_logado = None

def login(db):
    global usuario_logado
    email = input("Digite o e-mail: ")
    senha = getpass("Digite a senha: ")
    user_id = db.autenticar_usuario(email, senha)
    if user_id:
        print(f"Login realizado com sucesso! ID do usuário: {user_id}")
        usuario_logado = user_id  # atualiza a variável global
        return user_id
    else:
        print("Credenciais inválidas.")
        return None


if __name__ == "__main__":
    db = Database(dbname="sementesdb", user="postgres", password="123")

    usuario_logado = None
    while True:
        opcao = menu()
        if opcao == "1":
            cadastrar(db)
        elif opcao == "2":
            usuario_logado = login(db)
        elif opcao == "3":
            break
        else:
            print("Opção inválida.")

    db.fechar()
