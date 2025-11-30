import { PassThrough } from 'stream';

export const config = {
  api: {
    bodyParser: false, // importante para FormData
    sizeLimit: '10mb',
  },
};

export default async function handler(req, res) {
  const url = `http://18.216.31.10:8000${req.url.replace('/api/proxy8000', '')}`;

  const headers = { ...req.headers };
  delete headers.host;

  const response = await fetch(url, {
    method: req.method,
    headers,
    body: req.method !== 'GET' && req.method !== 'HEAD' ? req : undefined,
  });

  const contentType = response.headers.get('content-type') || 'application/json';
  res.status(response.status).setHeader('Content-Type', contentType);

  response.body.pipe(res);
}
