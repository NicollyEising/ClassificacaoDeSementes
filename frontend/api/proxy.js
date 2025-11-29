import fetch from 'node-fetch';

export const config = {
  api: {
    bodyParser: false, // necessário para uploads de arquivos
  },
};

export default async function handler(req, res) {
  if (req.method === 'POST') {
    try {
      const response = await fetch('http://18.216.31.10:8000/processar_imagem', {
        method: 'POST',
        headers: req.headers, // mantém os headers do cliente
        body: req, // encaminha o corpo da requisição
      });

      const data = await response.text(); // ou .json() se a API retornar JSON
      res.status(response.status).send(data);
    } catch (error) {
      res.status(500).json({ error: 'Erro ao acessar o backend' });
    }
  } else {
    res.status(405).json({ error: 'Método não permitido' });
  }
}
