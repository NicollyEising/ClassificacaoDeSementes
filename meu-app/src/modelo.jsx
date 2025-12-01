import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './index.css';
import './style.css';

export default function TrainingSummary() {
  const navigate = useNavigate();
  const [sidebarAberto, setSidebarAberto] = useState(true);

  const toggleSidebar = () => setSidebarAberto(prev => !prev);

  const handleLogout = () => {
    localStorage.removeItem('usuario_logado');
    sessionStorage.removeItem('usuario_logado');
    navigate('/login', { replace: true });
  };

  // Dados do resumo (conforme solicitado)
  const resumo = {
    titulo: 'Resumo do Treinamento de Classificação de Sementes',
    data_hora: '2025-10-01 07:18:57',
    melhor_acuracia_validacao: 0.9997,
    f1_weighted: 0.9993,
    iou_medio: 0.9986,
    epocas_executadas: 21,
    relatorio_por_classe: [
      { nome: 'Broken_soybeans', precision: 1.000, recall: 0.997, f1: 0.998 },
      { nome: 'Immature_soybeans', precision: 1.000, recall: 1.000, f1: 1.000 },
      { nome: 'Intact_soybeans', precision: 1.000, recall: 1.000, f1: 1.000 },
      { nome: 'Skin-damaged_soybeans', precision: 1.000, recall: 1.000, f1: 1.000 },
      { nome: 'Spotted_soybeans', precision: 0.997, recall: 1.000, f1: 0.998 },
      { nome: 'macro avg', precision: 0.999, recall: 0.999, f1: 0.999 },
      { nome: 'weighted avg', precision: 0.999, recall: 0.999, f1: 0.999 }
    ],
    distribuicao_amostras: { treino: 12000, validacao: 3000 }
  };

  const copiaTextoResumo = () => {
    const linhas = [];
    linhas.push(resumo.titulo);
    linhas.push('Data/Hora: ' + resumo.data_hora);
    linhas.push('Melhor Acurácia Validação: ' + resumo.melhor_acuracia_validacao);
    linhas.push('F1-Weighted: ' + resumo.f1_weighted);
    linhas.push('IoU médio: ' + resumo.iou_medio);
    linhas.push('Épocas executadas: ' + resumo.epocas_executadas);
    linhas.push('\nRelatório por classe:');
    resumo.relatorio_por_classe.forEach(c => {
      linhas.push(`- ${c.nome} | precision=${c.precision.toFixed(3)} recall=${c.recall.toFixed(3)} f1=${c.f1.toFixed(3)}`);
    });
    linhas.push('\nDistribuição de amostras:');
    linhas.push(`Treino: ${resumo.distribuicao_amostras.treino}, Validação: ${resumo.distribuicao_amostras.validacao}`);

    navigator.clipboard?.writeText(linhas.join('\n'))
      .then(() => alert('Resumo copiado para a área de transferência.'))
      .catch(() => alert('Não foi possível copiar.'));
  };

  return (
    <div className="ui grid">
      {/* Sidebar */}
      <div className={`four wide column sidebar-column ${sidebarAberto ? '' : 'hidden'}`} id="sidebar">
        <div className="ui vertical menu full-height" id="menu">
          <div className="menu-content">
            <div className="item mt-5"></div>
            <div className="item mt-5">
              <div className="menu">
                <Link className="item" to="/dashboard"><i className="chart line icon"></i>Dashboard</Link>
                <Link className="item" to="/dashboard#Defeitos"><i className="times circle outline icon"></i>Lista de Classificações</Link>
                <Link className="item" to="/input"><i className="boxes icon"></i>Enviar Arquivo</Link>
                <Link className="item" to="/modelo"><i className="book icon"></i>Modelo Utilizado</Link>
              </div>
            </div>
          </div>
          <div className="item profile-bottom">
            <div className="header">Support</div>
            <div className="menu">
              <a className="item horizontal" href="#perfil">
                <img className="ui mini circular image" src="https://semantic-ui.com/images/avatar2/small/molly.png" alt="Molly" />
                <div className="content profile-content">
                  <div className="ui sub header">Molly</div>
                  Coordinator
                </div>
              </a>
              <div className="item profile-bottom logout-item" style={{ marginTop: '1rem' }}>
                <button type="button" className="ui red button" style={{ width: '100%' }} onClick={handleLogout}>Sair</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Botão de alternância */}
      <button
        id="sidebar-toggle"
        className="ui icon button"
        style={{
          position: 'fixed',
          top: '20px',
          left: sidebarAberto ? '250px' : '20px',
          zIndex: 1100,
          transition: 'left 0.3s ease',
          background: 'white',
          border: '1px solid #ddd'
        }}
        onClick={toggleSidebar}
      >
        <i className={`chevron ${sidebarAberto ? 'left' : 'right'} icon`}></i>
      </button>

      {/* Conteúdo principal */}
      <div
        className="twelve wide column main-column"
        id="main-content"
        style={{ marginLeft: sidebarAberto ? '260px' : '0', padding: '2rem', transition: 'margin-left 0.3s ease' }}
      >
        <div id="training-container" className="item" style={{ marginTop: '1rem' }}>
          <div className="ui raised very padded text container segment transition scale-in" 
               style={{ maxWidth: '900px', marginTop: '2rem', borderRadius: '1rem' }}>
            <div className="flex flex-col space-y-4">
              <header className="mb-4">
                <h1 className="ui header">{resumo.titulo}</h1>
                <div className="text-gray-500 text-sm">
                  <i className="calendar icon" /> {new Date(resumo.data_hora).toLocaleString('pt-BR')}
                </div>
              </header>

              <section className="ui segment">
                <h3 className="ui header">Métricas gerais</h3>
                <div className="ui list">
                  <div className="item"><strong>Melhor Acurácia Validação:</strong> {resumo.melhor_acuracia_validacao}</div>
                  <div className="item"><strong>F1-Weighted:</strong> {resumo.f1_weighted}</div>
                  <div className="item"><strong>IoU médio:</strong> {resumo.iou_medio}</div>
                  <div className="item"><strong>Épocas executadas:</strong> {resumo.epocas_executadas}</div>
                </div>
              </section>

              <section className="ui segment">
                <h3 className="ui header">Relatório por classe</h3>
                <table className="ui celled table">
                  <thead>
                    <tr>
                      <th>Classe</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resumo.relatorio_por_classe.map((c, idx) => (
                      <tr key={idx}>
                        <td>{c.nome}</td>
                        <td>{c.precision.toFixed(3)}</td>
                        <td>{c.recall.toFixed(3)}</td>
                        <td>{c.f1.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              <section className="ui segment">
                <h3 className="ui header">Distribuição de amostras</h3>
                <p><strong>Treino:</strong> {resumo.distribuicao_amostras.treino}</p>
                <p><strong>Validação:</strong> {resumo.distribuicao_amostras.validacao}</p>
              </section>

              <div className="ui right aligned basic segment">
                <button className="ui button" onClick={copiaTextoResumo}>Copiar resumo</button>
                <a className="ui button primary" href="#" onClick={(e) => { e.preventDefault(); window.print(); }}>Imprimir</a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
