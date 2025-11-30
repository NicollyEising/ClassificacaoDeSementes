import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './index.css';
import './style.css';

function Input() {
  const navigate = useNavigate();

  useEffect(() => {
    'use strict';

    const qs = (selector, parent = document) => parent.querySelector(selector);
    const createElementFromHTML = (html) => {
      const template = document.createElement('template');
      template.innerHTML = html.trim();
      return template.content.firstChild;
    };
    const showMessage = (container, text, type = 'info') => {
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
    };

    const LOCAL_KEY = 'demo_users_v1';
    const getLocalUsers = () => {
      try { const raw = localStorage.getItem(LOCAL_KEY); return raw ? JSON.parse(raw) : []; } 
      catch { return []; }
    };
    const findLocalUser = (email, senha) => {
      const users = getLocalUsers();
      return users.find(u => u.email === email && u.senha === senha) || null;
    };

    const handleSubmit = async (event) => {
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
        const payload = { email, senha };
        let backendResponded = false;

        try {
          const res = await fetch('https://18.216.31.10:5000/login', {
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
            setTimeout(() => navigate('/dashboard', { replace: true }), 1000);
            return;
          } else {
            backendResponded = true;
            let text = 'Erro no servidor.';
            try { const j = await res.json(); text = j.error || j.mensagem || text; } catch {}
            showMessage(container, text, 'error');
          }
        } catch { backendResponded = false; }

        if (!backendResponded) {
          const user = findLocalUser(email, senha);
          if (!user) { showMessage(container, 'Credenciais inválidas (demo local).', 'error'); return; }
          showMessage(container, 'Login realizado localmente (modo demo).', 'success');
          const storage = (remember && typeof Storage !== 'undefined') ? localStorage : sessionStorage;
          storage.setItem('usuario_logado', JSON.stringify({ id: user.id, email }));
          setTimeout(() => navigate('/dashboard', { replace: true }), 1000);
        }
      } catch (err) {
        console.error(err);
        showMessage(container, 'Erro inesperado ao processar o login.', 'error');
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Enviar';
      }
    };

    const isUserLogged = (usuario) => {
      try {
        if (!usuario) return false;
        const obj = JSON.parse(usuario);
        return obj && obj.id && obj.email && typeof obj.id === 'string' && typeof obj.email === 'string' && obj.id.trim() !== '' && obj.email.trim() !== '';
      } catch { return false; }
    };

    const init = () => {
      const usuario = localStorage.getItem('usuario_logado') || sessionStorage.getItem('usuario_logado');
      if (isUserLogged(usuario)) {
        navigate('/dashboard', { replace: true });
        return;
      }

      const form = qs('form');
      if (!form) return;
      form.addEventListener('submit', handleSubmit);

      const first = qs('input, textarea', form);
      if (first) first.focus();
    };

    init();
  }, [navigate]);

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        background: '#f1f8f4',
      }}
    >
      <div
        className="ui segment rounded"
        style={{ width: '80%', maxWidth: '500px', padding: '2rem' }}
      >
        <h1
          style={{
            fontSize: '1.5em',
            display: 'flex',
            textAlign: 'center',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 600,
            marginBottom: '20px',
          }}
        >
          Faça o seu Login!
        </h1>

        <form className="space-y-6">
          <div>
            <label htmlFor="email" className="block mb-2 text-sm font-medium text-gray-900">
              Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5"
              placeholder="john.doe@empresa.com"
              required
            />
          </div>

          <div>
            <label htmlFor="senha" className="block mb-2 text-sm font-medium text-gray-900">
              Senha
            </label>
            <input
              type="password"
              id="senha"
              name="senha"
              className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5"
              placeholder="••••••••"
              required
            />
          </div>

          <div className="flex items-start mb-4">
            <input
              id="remember"
              type="checkbox"
              className="w-4 h-4 border border-gray-300 bg-gray-50"
              required
            />
            <label htmlFor="remember" className="ml-2 text-sm font-medium text-gray-900">
              Mantenha-me conectado
            </label>
          </div>

          <button
            type="submit"
            className="text-white bg-green-700 hover:bg-green-800 focus:ring-4 focus:outline-none focus:ring-green-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center w-full"
          >
            Enviar
          </button>
        </form>

        {/* Novo link para cadastro */}
        <p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.875rem' }}>
          Não é cadastrado?{' '}
          <span
            style={{ color: '#16a34a', cursor: 'pointer', textDecoration: 'underline' }}
            onClick={() => navigate('/cadastro')}
          >
            Cadastre-se já
          </span>
        </p>
      </div>
    </div>
  );
}

export default Input;
