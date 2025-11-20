(function () {
    'use strict';
  
    // --- Helpers ---
    function qs(selector, parent = document) {
      return parent.querySelector(selector);
    }
  
    function createElementFromHTML(html) {
      const template = document.createElement('template');
      template.innerHTML = html.trim();
      return template.content.firstChild;
    }
  
    function showMessage(container, text, type = 'info') {
      let existing = container.querySelector('.js-feedback');
      if (existing) existing.remove();
  
      const div = createElementFromHTML(`<div class="js-feedback mt-4 p-3 rounded text-sm"></div>`);
      div.classList.add(
        type === 'success' ? 'bg-green-50' :
        type === 'error' ? 'bg-red-50' : 'bg-gray-50'
      );
      div.textContent = text;
      container.appendChild(div);
      return div;
    }
  
    // --- LocalStorage fallback (modo demo) ---
    const LOCAL_KEY = 'demo_users_v1';
  
    function getLocalUsers() {
      try {
        const raw = localStorage.getItem(LOCAL_KEY);
        return raw ? JSON.parse(raw) : [];
      } catch (e) {
        return [];
      }
    }
  
    function findLocalUser(email, senha) {
      const users = getLocalUsers();
      return users.find(u => u.email === email && u.senha === senha) || null; // Comparação com senha pura
    }
  
    // --- Logout ---
    window.logout = function () {
      localStorage.removeItem('usuario_logado');
      sessionStorage.removeItem('usuario_logado');
      window.location.href = '/login.html';
    };
  
    // --- Submit handler ---
    async function handleSubmit(event) {
      event.preventDefault();
  
      const form = event.currentTarget;
      const container = form.closest('.ui.segment') || form;
      const email = qs('#email', form).value.trim();
      const senha = qs('#senha', form).value;
      const remember = qs('#remember', form).checked;
      const submitBtn = qs('button[type="submit"]', form);
  
      const old = container.querySelector('.js-feedback');
      if (old) old.remove();
  
      if (!email) { showMessage(container, 'Email obrigatório.', 'error'); return; }
      if (!senha) { showMessage(container, 'Senha obrigatória.', 'error'); return; }
  
      submitBtn.disabled = true;
      submitBtn.textContent = 'Enviando...';
  
      try {
        const payload = { email, senha }; // Senha enviada pura (sem hash)
        let backendResponded = false;
  
        try {
          const res = await fetch('http://127.0.0.1:5000/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
  
          if (res.ok) {
            backendResponded = true;
            const json = await res.json();
            showMessage(container, json.mensagem || 'Login realizado com sucesso.', 'success');
  
            const storage = (remember && typeof Storage !== 'undefined') ? localStorage : sessionStorage;
            storage.setItem('usuario_logado', JSON.stringify({ id: json.id, email }));
  
            // Redirecionar após login
            setTimeout(() => window.location.href = '/frontend/index.html', 1000);
            return;
          } else {
            backendResponded = true;
            let text = 'Erro no servidor.';
            try { const j = await res.json(); text = j.error || j.mensagem || text; } catch(e){}
            showMessage(container, text, 'error');
          }
        } catch {
          backendResponded = false;
        }
  
        // Fallback local (modo demo)
        if (!backendResponded) {
          const user = findLocalUser(email, senha); // Comparação com senha pura
          if (!user) {
            showMessage(container, 'Credenciais inválidas (demo local).', 'error');
            return;
          }
          showMessage(container, 'Login realizado localmente (modo demo).', 'success');
  
          const storage = (remember && typeof Storage !== 'undefined') ? localStorage : sessionStorage;
          storage.setItem('usuario_logado', JSON.stringify({ id: user.id, email }));
  
          setTimeout(() => window.location.href = '/dashboard.html', 1000);
        }
  
      } catch (err) {
        console.error(err);
        showMessage(container, 'Erro inesperado ao processar o login.', 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Enviar';
      }
    }
  
    // --- Verificação de usuário logado ---
    function isUserLogged(usuario) {
      try {
        if (!usuario) return false;
        const obj = JSON.parse(usuario);
        // Validação mais rigorosa: id e email devem ser strings não vazias
        return obj && obj.id && obj.email && typeof obj.id === 'string' && typeof obj.email === 'string' && obj.id.trim() !== '' && obj.email.trim() !== '';
      } catch {
        return false;
      }
    }
  
    // --- Inicialização ---
    function init() {
      const usuario = localStorage.getItem('usuario_logado') || sessionStorage.getItem('usuario_logado');
      if (isUserLogged(usuario)) {
        console.log('Usuário logado:', JSON.parse(usuario));
        window.location.href = '/frontend/index.html';
        return;
      } else {
        // Log de debug para confirmar que não há usuário logado
        console.log('Nenhum usuário logado encontrado. Permanecendo na página de login.');
      }
  
      const form = qs('form');
      if (!form) return;
      form.addEventListener('submit', handleSubmit);
  
      const first = qs('input, textarea', form);
      if (first) first.focus();
    }
  
    document.addEventListener('DOMContentLoaded', init);
  })();