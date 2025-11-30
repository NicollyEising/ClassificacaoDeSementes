const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();

// Proxy para backend 5000
app.use('/api', createProxyMiddleware({
  target: 'http://18.216.31.10:5000',
  changeOrigin: true,
  pathRewrite: { '^/api5000': '' }, // remove o prefixo na requisição
}));

// Proxy para backend 8000
app.use('/api8000', createProxyMiddleware({
  target: 'http://18.216.31.10:8000',
  changeOrigin: true,
  pathRewrite: { '^/api8000': '' },
}));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`Proxy rodando na porta ${PORT}`));
