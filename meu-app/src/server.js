// server.js
const express = require('express');
const path = require('path');
const morgan = require('morgan');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');

const app = express();
app.use(morgan('dev'));

// ---- CORS ----
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// ---- PROXIES ----
// Important: register proxies BEFORE any body-parser that would consume the request stream

// Proxy -> backend 5000
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

// Proxy -> backend 8000
app.use('/api8000', createProxyMiddleware({
  target: 'http://18.216.31.10:8000',
  changeOrigin: true,
  secure: false,
  logLevel: 'debug',
  pathRewrite: { '^/api8000': '' },
  onProxyReq: (proxyReq, req, res) => {
    // Garantir que multipart/form-data passe corretamente
    if (req.body) {
      const bodyData = JSON.stringify(req.body);
      proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
      proxyReq.write(bodyData);
    }
  },
  onError: (err, req, res) => {
    console.error('Proxy /api8000 error:', err?.message || err);
    if (!res.headersSent) res.status(502).json({ error: 'Bad gateway (api8000)', details: String(err) });
    else res.end();
  }
}));

// ---- Serve React build ----
const buildDir = path.join(__dirname, 'build');
app.use(express.static(buildDir));

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(buildDir, 'index.html'));
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
  console.log(`Proxying /api5000 -> http://18.216.31.10:5000`);
  console.log(`Proxying /api8000 -> http://18.216.31.10:8000`);
});
