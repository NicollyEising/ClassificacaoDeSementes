const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://18.216.31.10:5000',
      changeOrigin: true,
      pathRewrite: { '^/api': '' },
    })
  );

  app.use(
    '/api8000',
    createProxyMiddleware({
      target: 'http://18.216.31.10:8000',
      changeOrigin: true,
      pathRewrite: { '^/api8000': '' },
    })
  );
};
