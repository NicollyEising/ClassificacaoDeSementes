// server.js
const express = require('express');
const path = require('path');
const morgan = require('morgan');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
app.use(morgan('dev'));

// ---- CORS ----
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*'); 
  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

// ---- PROXIES ----
// Proxy backend 5000
app.use('/api5000', createProxyMiddleware({
  target: 'http://18.216.31.10:5000',
  changeOrigin: true,
  secure: false,
  logLevel: 'debug',
  pathRewrite: { '^/api5000': '' },
  onError: (err, req, res) => {
    console.error('Proxy /api5000 error:', err?.message || err);
    if (!res.headersSent) res.status(502).json({ error: 'Bad gateway (api5000)', details: String(err) });
    else res.end();
  }
}));

// Proxy backend 8000 (para uploads e JSON)
app.use('/api8000', createProxyMiddleware({
  target: 'http://18.216.31.10:8000',
  changeOrigin: true,
  secure: false,
  logLevel: 'debug',
  pathRewrite: { '^/api8000': '' },
  onError: (err, req, res) => {
    console.error('Proxy /api8000 error:', err?.message || err);
    if (!res.headersSent) res.status(502).json({ error: 'Bad gateway (api8000)', details: String(err) });
    else res.end();
  },
  // IMPORTANTE: permite enviar arquivos sem quebrar stream
  selfHandleResponse: false
}));

// ---- Serve React build ----
const buildDir = path.join(__dirname, 'build');
app.use(express.static(buildDir));
app.get('*', (req, res) => {
  res.sendFile(path.join(buildDir, 'index.html'));
});

// ---- Start server ----
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log(`Proxying /api5000 -> http://18.216.31.10:5000`);
  console.log(`Proxying /api8000 -> http://18.216.31.10:8000`);
});
