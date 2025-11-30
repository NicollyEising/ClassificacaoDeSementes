// pages/api/proxy8000/[...path].js
export default async function handler(req, res) {
  try {
    const url = `http://18.216.31.10:8000${req.url}`;

    const response = await fetch(url, {
      method: req.method,
      headers: {
        ...req.headers,
        host: '18.216.31.10:8000',
      },
      body: req.method !== 'GET' && req.method !== 'HEAD' ? req : undefined,
    });

    // Lê a resposta como text para evitar body already read
    const data = await response.text();

    res
      .status(response.status)
      .setHeader('Content-Type', response.headers.get('content-type') || 'text/plain')
      .send(data);
  } catch (error) {
    res.status(500).json({ detail: error.message });
  }
}

export const config = {
  api: {
    bodyParser: false, // Importante para upload de arquivos/FormData
    sizeLimit: '10mb',
  },
};
