// cadastro.js
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
      div.classList.add(type === 'success' ? 'bg-green-50 text-green-700' : 
                        type === 'error' ? 'bg-red-50 text-red-700' : 'bg-gray-50 text-gray-700');
      div.textContent = text;
      container.appendChild(div);
      return div;
    }
  
    function validateEmail(email) {
      return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }
  
    function minPasswordOk(password) {
      return password && password.length >= 6;
    }
  
    // --- Submit handler ---
    async function handleSubmit(event) {
      event.preventDefault();
  
      const form = event.currentTarget;
      const container = form.closest('.ui.segment') || form;
      const email = qs('#email', form).value.trim();
      const senha = qs('#senha', form).value;
      const remember = qs('#remember', form)?.checked ?? true;
      const submitBtn = qs('button[type="submit"]', form);
  
      // Limpar mensagens antigas
      const old = container.querySelector('.js-feedback');
      if (old) old.remove();
  
      // Validações
      if (!email || !validateEmail(email)) { showMessage(container, 'Email inválido.', 'error'); return; }
      if (!minPasswordOk(senha)) { showMessage(container, 'Senha deve ter ao menos 6 caracteres.', 'error'); return; }
      if (!remember) { showMessage(container, 'É preciso concordar com os termos e condições.', 'error'); return; }
  
      // Desabilitar botão
      submitBtn.disabled = true;
      submitBtn.textContent = 'Enviando...';
  
      try {
        // Envia senha pura para o backend
        const payload = { email, senha };
  
        const res = await fetch('http://127.0.0.1:5000/cadastrar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
  
        if (!res.ok) {
          let text = `Erro ${res.status} no servidor`;
          try { const j = await res.json(); text = j.erro || j.mensagem || text; } catch(e){}
          showMessage(container, text, 'error');
          return;
        }
  
        const json = await res.json();
        if (json.id) {
          // Usuário cadastrado com sucesso, ID gerado pelo backend
          showMessage(container, json.mensagem || 'Cadastro realizado com sucesso.', 'success');
          sessionStorage.setItem('usuario_logado', JSON.stringify({ id: json.id, email }));
          if (json.redirect) window.location.href = json.redirect;
        } else {
          showMessage(container, json.erro || 'Erro ao cadastrar usuário.', 'error');
        }
  
      } catch (err) {
        console.error(err);
        showMessage(container, 'Erro inesperado ao processar o cadastro.', 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Enviar';
      }
    }
  
    // --- Inicialização ---
    function init() {
      const form = qs('form');
      if (!form) return;
      form.addEventListener('submit', handleSubmit);
  
      // Focar no primeiro input
      const first = qs('input, textarea', form);
      if (first) first.focus();
    }
  
    document.addEventListener('DOMContentLoaded', init);
})();
