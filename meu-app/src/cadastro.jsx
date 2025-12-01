import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function Cadastro() {
  const [email, setEmail] = useState('');
  const [senha, setSenha] = useState('');
  const [remember, setRemember] = useState(true);
  const [feedback, setFeedback] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  // Redireciona se já houver usuário logado
  useEffect(() => {
    const usuarioLogado = JSON.parse(localStorage.getItem('usuario_logado'));
    if (usuarioLogado?.id) {
      navigate('/dashboard', { replace: true });
    }
  }, [navigate]);

  const estilos = {
    htmlBody: { height: '100%', margin: 0, fontFamily: 'sans-serif' },
    centerWrapper: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      background: '#f1f8f4',
      padding: '1rem',
    },
    segment: {
      width: '100%',
      maxWidth: '500px',
      padding: '2rem',
      borderRadius: '1rem',
      background: 'white',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
    },
    header: { fontSize: '1.5em', textAlign: 'center', fontWeight: 600, marginBottom: '20px' },
    input: {
      width: '100%',
      padding: '0.625rem',
      borderRadius: '0.5rem',
      border: '1px solid #d1d5db',
      backgroundColor: '#f9fafb',
      marginBottom: '1rem',
    },
    checkboxWrapper: { display: 'flex', alignItems: 'center', marginBottom: '1rem' },
    checkboxLabel: { marginLeft: '0.5rem', fontSize: '0.875rem', color: '#111827' },
    button: {
      width: '100%',
      padding: '0.625rem',
      borderRadius: '0.5rem',
      backgroundColor: '#16a34a',
      color: 'white',
      fontWeight: 500,
      fontSize: '0.875rem',
      cursor: 'pointer',
      border: 'none',
    },
    link: { color: '#16a34a', textDecoration: 'underline' },
    feedback: { marginTop: '1rem', padding: '0.75rem 1rem', borderRadius: '0.5rem', fontSize: '0.875rem' },
  };

  const validateEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const minPasswordOk = (password) => password && password.length >= 6;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFeedback(null);

    if (!validateEmail(email)) return setFeedback({ text: 'Email inválido.', type: 'error' });
    if (!minPasswordOk(senha)) return setFeedback({ text: 'Senha deve ter ao menos 6 caracteres.', type: 'error' });
    if (!remember) return setFeedback({ text: 'É preciso concordar com os termos e condições.', type: 'error' });

    setSubmitting(true);

    try {
      const payload = { email, senha };
      const res = await fetch('https://api.sementes.lat:5000/cadastrar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const json = await res.json();

      if (!res.ok) {
        setFeedback({ text: json.erro || json.mensagem || `Erro ${res.status}`, type: 'error' });
      } else if (json.id) {
        setFeedback({ text: json.mensagem || 'Cadastro realizado com sucesso.', type: 'success' });
        // salva no localStorage
        localStorage.setItem('usuario_logado', JSON.stringify({ id: json.id, email }));
        // redireciona para dashboard
        navigate('/dashboard', { replace: true });
      } else {
        setFeedback({ text: json.erro || 'Erro ao cadastrar usuário.', type: 'error' });
      }
    } catch (err) {
      console.error(err);
      setFeedback({ text: 'Erro inesperado ao processar o cadastro.', type: 'error' });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={estilos.htmlBody}>
      <div style={estilos.centerWrapper}>
        <div style={estilos.segment}>
          <h1 style={estilos.header}>Faça o seu Cadastro!</h1>
          <form onSubmit={handleSubmit}>
  <div>
    <label htmlFor="email">Email</label>
    <input
      type="email"
      id="email"
      name="email"
      placeholder="john.doe@empresa.com"
      required
      value={email}
      onChange={(e) => setEmail(e.target.value)}
      style={estilos.input}
    />
  </div>
  <div>
    <label htmlFor="senha">Senha</label>
    <input
      type="password"
      id="senha"
      name="senha"
      placeholder="••••••••"
      required
      value={senha}
      onChange={(e) => setSenha(e.target.value)}
      style={estilos.input}
    />
  </div>
  <div style={estilos.checkboxWrapper}>
    <input id="remember" type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
    <label htmlFor="remember" style={estilos.checkboxLabel}>
      Eu concordo com os{' '}
      <a href="#" style={estilos.link}>
        termos e condições
      </a>.
    </label>
  </div>
  <button type="submit" style={estilos.button} disabled={submitting}>
    {submitting ? 'Enviando...' : 'Enviar'}
  </button>
</form>

{/* Novo link para login */}
<p style={{ textAlign: 'center', marginTop: '1rem', fontSize: '0.875rem' }}>
  Já tem uma conta?{' '}
  <span
    style={{ color: '#16a34a', cursor: 'pointer', textDecoration: 'underline' }}
    onClick={() => navigate('/login')}
  >
    Faça login
  </span>
</p>

{feedback && (
  <div
    style={{
      ...estilos.feedback,
      backgroundColor: feedback.type === 'success' ? '#dcfce7' : '#fee2e2',
      color: feedback.type === 'success' ? '#166534' : '#991b1b',
    }}
  >
    {feedback.text}
  </div>
)}
        </div>
      </div>
    </div>
  );
}

export default Cadastro;
