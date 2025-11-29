import boto3
import os

# Configuração do acesso ao Contabo Object Storage
s3 = boto3.client(
    's3',
    endpoint_url='https://usc1.contabostorage.com',  # endpoint da sua região
    aws_access_key_id='5bafdce4ac57085c40c8c59ebaf58c5e',
    aws_secret_access_key='061355d0315ceeca4f509efd28189a4e'
)

bucket_name = 'teste'          # nome do bucket
local_dir = '/tmp/bucket_files'  # pasta local para salvar os arquivos

# Cria a pasta local se não existir
os.makedirs(local_dir, exist_ok=True)

# Lista todos os objetos no bucket
objetos = s3.list_objects_v2(Bucket=bucket_name)

# Baixa todos os arquivos
for obj in objetos.get('Contents', []):
    arquivo = obj['Key']
    caminho_local = os.path.join(local_dir, arquivo.replace('/', os.sep))  # preserva subpastas

    # Cria pastas locais se necessário
    os.makedirs(os.path.dirname(caminho_local), exist_ok=True)

    print(f"Baixando {arquivo} para {caminho_local} ...")
    s3.download_file(bucket_name, arquivo, caminho_local)

print("Download concluído!")
