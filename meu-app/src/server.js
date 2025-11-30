const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

// Proxy para backend 5000
app.use('/api', createProxyMiddleware({
  target: 'http://18.216.31.10:5000',
  changeOrigin: true,
  pathRewrite: { '^/api': '' }
}));

// Proxy para backend 8000
app.use('/api8000', createProxyMiddleware({
  target: 'http://18.216.31.10:8000',
  changeOrigin: true,
  pathRewrite: { '^/api8000': '' }
}));

// Serve os arquivos do build do React
app.use(express.static(path.join(__dirname, 'build')));

// Todas as outras rotas servem o React
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Servidor rodando na porta ${PORT}`));
