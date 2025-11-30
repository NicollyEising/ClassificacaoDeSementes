import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import './index.css';
import './style.css';

function Input() {
  const navigate = useNavigate();
  const location = useLocation();

  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [fileName, setFileName] = useState('');

  const toggleSidebar = () => setSidebarHidden(!sidebarHidden);

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFileName(e.target.files[0].name);
    } else {
      setFileName('');
    }
  };

  
      // Função para logout
      const handleLogout = () => {
        localStorage.removeItem("usuario_logado");
        sessionStorage.removeItem("usuario_logado");
        navigate("/login", { replace: true });
      };

  // Scroll para hash quando a rota mudar
  useEffect(() => {
    if (location.hash) {
      const el = document.querySelector(location.hash);
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  }, [location]);

  // Script adaptado do DOMContentLoaded
  useEffect(() => {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.getElementById('sidebar-toggle');
    const fileInput = document.getElementById('fileInput');
    const fileNameInput = document.getElementById('file-name');
    const enviarBtn = document.getElementById('enviarBtn');

    if (!toggleButton || !sidebar) {
      console.error("toggleButton ou sidebar inexistente. Verifique os IDs.");
      return;
    }

    const toggleHandler = () => {
      sidebar.classList.toggle('hidden');
      const icon = toggleButton.querySelector('i');
      if (!icon) return;
      if (sidebar.classList.contains('hidden')) {
        icon.classList.remove('left');
        icon.classList.add('right');
      } else {
        icon.classList.remove('right');
        icon.classList.add('left');
      }
    };

    toggleButton.addEventListener('click', toggleHandler);

    const fileChangeHandler = () => {
      const fName = fileInput.files.length ? fileInput.files[0].name : 'Nenhum arquivo selecionado';
      setFileName(fName);
      if (fileNameInput) fileNameInput.value = fName;
    };

    if (fileInput) fileInput.addEventListener('change', fileChangeHandler);

    const enviarHandler = async () => {
      const cidadeInput = document.getElementById('cidade');
      const cidade = cidadeInput ? cidadeInput.value || 'Jaragua do Sul' : 'Jaragua do Sul';

      if (!fileInput || !fileInput.files.length) {
        alert('Selecione uma imagem');
        return;
      }

      const usuarioData = localStorage.getItem('usuario_logado');
      if (!usuarioData) {
        alert('Usuário não encontrado no localStorage');
        return;
      }

      const usuario = JSON.parse(usuarioData);
      const usuarioId = usuario.id;

      const formData = new FormData();
      formData.append('arquivo', fileInput.files[0]);
      formData.append('cidade', cidade);
      formData.append('usuario_id', usuarioId);

      try {
        const response = await fetch('/api8000/processar_imagem', {
          method: 'POST',
          body: formData
        });

        // Lê a resposta apenas uma vez
        const data = await response.json().catch(async () => {
          const text = await response.text();
          throw new Error(`Resposta inválida do servidor: ${text}`);
        });

        if (!response.ok) {
          throw new Error(data.detail || 'Erro ao processar imagem');
        }

        const resultadoDiv = document.getElementById('resultado');
        if (!resultadoDiv) return;

        resultadoDiv.classList.remove('hidden');
        resultadoDiv.innerHTML = `
          <p><strong>Classe prevista:</strong> ${data.classe_prevista}</p>
          <p><strong>Probabilidade:</strong> ${data.probabilidade}</p>
          <p><strong>Cidade:</strong> ${data.clima.cidade}</p>
          <p><strong>Temperatura:</strong> ${data.clima.temperatura}°C</p>
          <p><strong>Condição:</strong> ${data.clima.condicao}</p>
          <p><strong>Chance de chuva:</strong> ${data.clima.chance_chuva}%</p>
          <img src="data:image/png;base64,${data.imagem_anotada_base64}"
               alt="Imagem Anotada"
               style="max-width:400px; margin-top:1rem;"/>
        `;
      } catch (error) {
        alert(error.message);
      }
    };

    if (enviarBtn) enviarBtn.addEventListener('click', enviarHandler);

    return () => {
      toggleButton.removeEventListener('click', toggleHandler);
      if (fileInput) fileInput.removeEventListener('change', fileChangeHandler);
      if (enviarBtn) enviarBtn.removeEventListener('click', enviarHandler);
    };
  }, []);

  return (
    <div>
      {/* Estilos internos */}
      <style>{`
        .hidden { display: none !important; }
        #sidebar {
          position: fixed; top: 0; left: 0; width: 250px; height: 100vh; padding-top: 1rem;
          transition: transform 0.3s ease; z-index: 1000;
        }
        .main-column { margin-left: 260px; padding: 2rem; transition: margin-left 0.3s ease; }
        #sidebar-toggle {
          position: fixed; top: 20px; left: 230px; z-index: 1100; transition: left 0.3s ease;
          background: white; border: 1px solid #ddd;
        }
        #sidebar.hidden { transform: translateX(-100%); }
        #sidebar.hidden + #sidebar-toggle { left: 20px; }
        #sidebar.hidden ~ .main-column { margin-left: 60px; }
      `}</style>

      <div className="ui grid">
        {/* Sidebar */}
        <div className={`four wide column sidebar-column ${sidebarHidden ? 'hidden' : ''}`} id="sidebar">
          <div className="ui vertical menu full-height" id="menu">
            <div className="menu-content">
              <div className="item mt-5"></div>
              <div className="item mt-5">
                <div className="menu">
                  <Link className="item" to="/dashboard">
                    <i className="chart line icon"></i>Dashboard
                  </Link>
                  <Link className="item" to="/dashboard#Defeitos">
                    <i className="times circle outline icon"></i>Lista de Classificações
                  </Link>
                  <Link className="item" to="/input">
                    <i className="boxes icon"></i>Enviar Arquivo
                  </Link>
                </div>
              </div>
            </div>

            <div className="item profile-bottom">
              <div className="header">Support</div>
              <div className="menu">
                <a className="item horizontal" href="#perfil">
                  <img className="ui mini circular image" src="https://semantic-ui.com/images/avatar2/small/molly.png" alt="Avatar" />
                  <div className="content profile-content">
                    <div className="ui sub header">Molly</div>
                    Coordinator
                  </div>
                </a>
                <div className="item profile-bottom logout-item" style={{ marginTop: '1rem' }}>
        <button
          type="button"
          className="ui red button"
          style={{ width: '100%' }}
          onClick={handleLogout}
        >
          Sair
        </button>
      </div>
              </div>
            </div>
          </div>
        </div>

        {/* Botão fixo de alternância */}
        <button
          id="sidebar-toggle"
          className="ui icon button"
          style={{ left: sidebarHidden ? '20px' : '230px' }}
          onClick={toggleSidebar}
        >
          <i className={`chevron ${sidebarHidden ? 'right' : 'left'} icon`}></i>
        </button>

        {/* Conteúdo principal */}
        <div className="twelve wide column main-column" style={{ marginTop: '2rem', marginLeft: sidebarHidden ? '60px' : '260px' }}>
          <div className="ui segment rounded shadow-md p-6 bg-white">
            <h3 className="ui header text-gray-700">
              Envio de Arquivo <span aria-hidden="true" className="text-green-500">●</span>
            </h3>

            <form className="ui form" style={{ marginTop: '1.5rem' }}>
              <div className="field">
                <label htmlFor="cidade" className="font-medium text-gray-600">Cidade</label>
                <div className="ui left icon input w-full">
                  <i className="map marker alternate icon"></i>
                  <input type="text" id="cidade" placeholder="Digite a cidade" className="rounded-md focus:ring-2 focus:ring-green-500" />
                </div>
              </div>

              <div className="field" style={{ marginTop: '1rem' }}>
                <label htmlFor="fileInput" className="font-medium text-gray-600">Selecione uma imagem</label>
                <div className="ui action input w-full">
                  <input type="text" id="file-name" value={fileName || 'Nenhum arquivo selecionado'} readOnly />
                  <label htmlFor="fileInput" className="ui button gray">Escolher arquivo</label>
                  <input type="file" id="fileInput" accept="image/*" className="hidden" onChange={handleFileChange} />
                </div>
              </div>

              <div className="field" style={{ marginTop: '1.5rem' }}>
                <button type="button" id="enviarBtn" className="ui green button fluid">
                  <i className="cloud upload icon"></i> Enviar Arquivo
                </button>
              </div>
            </form>

            <div id="resultado" className={`mt-6 p-4 bg-gray-50 rounded-md border border-gray-200 ${fileName ? '' : 'hidden'}`}></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Input;




