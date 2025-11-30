import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Chart from 'react-apexcharts';
import './index.css';
import './style.css';

function MeuComponente() {
  const navigate = useNavigate();
  const location = useLocation();

  const [secaoAtiva, setSecaoAtiva] = useState('rails');
  const [sidebarHidden, setSidebarHidden] = useState(false);
  const [resultadosDefeitos, setResultadosDefeitos] = useState([]);
  const backendURL = "/api";
  const [chartData, setChartData] = useState({
    series: [],
    options: {}
  });


      // Função para logout
      const handleLogout = () => {
        localStorage.removeItem("usuario_logado");
        sessionStorage.removeItem("usuario_logado");
        navigate("/login", { replace: true });
      };
  
      const [classeFiltro, setClasseFiltro] = useState(null); // estado para armazenar a classe selecionada
      const [dropdownAberto, setDropdownAberto] = useState(false);

// Filtra os resultados conforme a classe selecionada
  const resultadosFiltrados = classeFiltro
  ? resultadosDefeitos.filter(r => r.classe_prevista === classeFiltro)
  : resultadosDefeitos;


      useEffect(() => {
        if (resultadosDefeitos.length === 0) return;
      
        const ultimosResultados = resultadosDefeitos.slice(-9); // últimos 9
      
        setChartData({
          series: [
            {
              name: 'Probabilidade de Crescimento',
              data: ultimosResultados.map(r => (probabilidadeCrescimento(r) * 100).toFixed(2))
            }
          ],
          options: {
            chart: { type: 'area', height: 350 },
            xaxis: {
              categories: ultimosResultados.map(r => new Date(r.data_hora).toLocaleDateString())
            },
            colors: ['rgb(55, 208, 75)'],
            fill: { type: 'solid', opacity: 0.3, colors: ['rgb(55, 208, 75)'] }
          }
        });
      }, [resultadosDefeitos]);

  // Fetch de resultados do backend
  useEffect(() => {

    const usuarioLogado = JSON.parse(localStorage.getItem("usuario_logado"));

    if (!usuarioLogado || !usuarioLogado.id) {
      navigate('/login', { replace: true }); // redireciona imediatamente
      return;
    }



    const fetchResultados = async () => {
      try {
        const usuarioLogado = JSON.parse(localStorage.getItem("usuario_logado"));
        if (!usuarioLogado || !usuarioLogado.id) return;
        const usuarioId = usuarioLogado.id;

        const res = await fetch(`${backendURL}/resultados/${usuarioId}`);
        if (!res.ok) throw new Error("Nenhum resultado encontrado para este usuário");
        const resultados = await res.json();

        // Mantém a imagem do backend ou fallback
        const resultadosComImagem = resultados.map(r => ({
          ...r,
          imagem: r.imagem || null
        }));

        setResultadosDefeitos(resultadosComImagem);
      } catch (err) {
        console.error(err);
      }
    };

    fetchResultados();
  }, []);

  // Sincroniza hash da URL com seção ativa
  useEffect(() => {
    if (location.hash) {
      const hash = location.hash.replace('#', '');
      setSecaoAtiva(hash);
      setTimeout(() => {
        const element = document.getElementById(hash);
        if (element) element.scrollIntoView({ behavior: 'smooth' });
      }, 0);
    } else {
      setSecaoAtiva('rails');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [location]);

  const abrirSecao = (nome) => {
    if (nome === 'Recomendacoes') {
      navigate('/input', { replace: false }); // caminho absoluto
    } else if (nome === 'Defeitos') {
      navigate('/dashboard#Defeitos', { replace: false }); // caminho absoluto
    } else {
      setSecaoAtiva(nome);
      navigate('/dashboard', { replace: false }); // caminho absoluto
    }
  };

  const displayStyle = (nome) => ({
    display: secaoAtiva === nome ? 'block' : 'none',
  });

  const toggleSidebar = () => setSidebarHidden(!sidebarHidden);

  // Funções de cálculo
  const probabilidadeCrescimento = (registro) => {
    let pSemente = registro.classe_prevista === "Intact soybeans" ? 0.9 :
                    registro.classe_prevista === "Skin-damaged soybeans" ? 0.6 : 0.5;

    let pCondicao = registro.condicao === "Sol" ? 0.95 :
                    registro.condicao === "Nublado" ? 0.8 :
                    registro.condicao === "Chuva" ? 0.85 : 0.8;

    let pChuva = registro.chance_chuva > 0 ? 1 : 0.8;

    return pSemente * pCondicao * pChuva;
  };

  const taxaGeralCrescimento = (registros) => {
    if (!registros || registros.length === 0) return 0;
    const soma = registros.reduce((acc, r) => acc + probabilidadeCrescimento(r), 0);
    return (soma / registros.length * 100).toFixed(2);
  };

  const climaMaisFrequente = (registros) => {
    const contagem = {};
    registros.forEach(r => contagem[r.condicao] = (contagem[r.condicao] || 0) + 1);
    return Object.entries(contagem).reduce((a,b) => b[1] > a[1] ? b : a, ["",0])[0];
  };

  const classeMaisPredominante = (registros) => {
    const contagem = {};
    registros.forEach(r => contagem[r.classe_prevista] = (contagem[r.classe_prevista] || 0) + 1);
    return Object.entries(contagem).reduce((a,b) => b[1] > a[1] ? b : a, ["",0])[0];
  };

  const calcularDadosCirculares = (registros) => {
    const total = registros.length;
    if (total === 0) return { defeitos: 0, intact: 0, crescimento: 0 };

    const intactCount = registros.filter(r => r.classe_prevista === "Intact soybeans").length;
    const naoIntactCount = total - intactCount;
    const taxaCrescimento = taxaGeralCrescimento(registros);

    return {
      defeitos: ((naoIntactCount / total) * 100).toFixed(2),
      intact: ((intactCount / total) * 100).toFixed(2),
      crescimento: taxaCrescimento
    };
  };

  // Dados para indicadores circulares
  const dadosCirculares = calcularDadosCirculares(resultadosDefeitos);
  const climaComum = climaMaisFrequente(resultadosDefeitos);
  const classePredominante = classeMaisPredominante(resultadosDefeitos);
  const taxaCrescimento = taxaGeralCrescimento(resultadosDefeitos);

  return (
    <div className="ui grid">
      {/* Sidebar */}
      <div className={`four wide column sidebar-column ${sidebarHidden ? 'hidden' : ''}`} id="sidebar">
        <div className="ui vertical menu full-height" id="menu">
          <div className="menu-content">
            <div className="item mt-5"></div>

            <div className="item mt-5">
              <div className="menu">
                <button
                  type="button"
                  className={`item ${secaoAtiva === 'rails' ? 'active' : ''}`}
                  onClick={() => abrirSecao('rails')}
                >
                  <i className="chart line icon"></i>Dashboard
                </button>
                <button
                  type="button"
                  className={`item ${secaoAtiva === 'Defeitos' ? 'active' : ''}`}
                  onClick={() => abrirSecao('Defeitos')}
                >
                  <i className="times circle outline icon"></i>Lista de Classificações
                </button>
                <button
                  type="button"
                  className={`item ${secaoAtiva === 'Recomendacoes' ? 'active' : ''}`}
                  onClick={() => abrirSecao('Recomendacoes')}
                >
                  <i className="boxes icon"></i>Enviar Arquivo
                </button>
              </div>
            </div>
          </div>

          <div className="item profile-bottom">
            <div className="header">Support
              
            </div>

            <div className="menu">
              <a className="item horizontal" href="#perfil">
                <img
                  className="ui mini circular image"
                  src="https://semantic-ui.com/images/avatar2/small/molly.png"
                  alt="Molly"
                />
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

      {/* Botão de alternância da sidebar */}
      <button
        id="sidebar-toggle"
        className="ui icon button"
        style={{ position: 'fixed', top: '20px', left: sidebarHidden ? '20px' : '230px', zIndex: 1100 }}
        onClick={toggleSidebar}
      >
        <i className={`chevron ${sidebarHidden ? 'right' : 'left'} icon`}></i>
      </button>

      <div className="main-column">
        <div className="twelve wide column">
          {/* SEÇÃO: Dashboard (rails) */}
          <div id="rails" className="ui content-section" style={displayStyle('rails')}>
            <div className="ui grid equal-height">
              <div className="eight wide column">
                <div className="ui rounded segment">
                <h3 className="titulo2" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
  Últimos Detectados
  <span style={{ position: 'relative' }}>
    <button 
      onClick={() => setDropdownAberto(!dropdownAberto)}
      style={{
        background: 'transparent',
        border: 'none',
        padding: '4px 8px',
        cursor: 'pointer',
        fontSize: '14px',
        color: '#4b5563', // cinza escuro
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
      }}
    >
      Filtrar por Gravidade <i className="angle down icon"></i>
    </button>

    {dropdownAberto && (
      <div
        className="menu"
        style={{
          position: 'absolute',
          top: '100%',
          right: 0,
          minWidth: '140px',
          background: 'white',
          borderRadius: '6px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          padding: '6px 0',
          zIndex: 1000,
        }}
      >
        {[...new Set(resultadosDefeitos.map(r => r.classe_prevista))].map(classe => (
          <div
            key={classe}
            className="item"
            onClick={() => { setClasseFiltro(classe); setDropdownAberto(false); }}
            style={{
              padding: '6px 10px',
              cursor: 'pointer',
              transition: 'background 0.2s',
            }}
            onMouseEnter={e => e.currentTarget.style.background = '#f0f0f0'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            {classe}
          </div>
        ))}
        <div
          className="item"
          onClick={() => { setClasseFiltro(null); setDropdownAberto(false); }}
          style={{
            padding: '6px 10px',
            cursor: 'pointer',
            fontWeight: 'bold',
            transition: 'background 0.2s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = '#f0f0f0'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          Todas
        </div>
      </div>
    )}
  </span>
</h3>


                  {/* Lista de defeitos */}
 {/* Lista de defeitos */}
 <div className="ui relaxed divided list">
  {resultadosFiltrados.slice(0, 9).map((resultado) => (
    <div
      key={resultado.id}
      className="lista"
      onClick={() => navigate(`/item/${resultado.id}`)}
    >
      <img
        src={resultado.imagem ? `data:image/png;base64,${resultado.imagem}` : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg"}
        alt={resultado.classe_prevista}
      />
      <div className="content">
        <span className="header">{resultado.classe_prevista}</span>
        <div className="description">
          Probabilidade: {(resultado.probabilidade*100).toFixed(2)}% | Cidade: {resultado.cidade} | Condição: {resultado.condicao}
        </div>
      </div>
    </div>
  ))}
</div>



                  <button
                    className="todosD ui button"
                    type="button"
                    onClick={() => abrirSecao('Defeitos')}
                  >
                    Todos os defeitos <i className="arrow right icon"></i>
                  </button>
                </div>
              </div>

              <div className="eight wide column">
                <div className="ui rounded segment">
                  <div className="ui grid">
                    <div className="column">
                      <div className="ui">
                        {/* grafico */}
                        <div className="w-full bg-white dark:bg-gray-800 p-2 md:p-3">
  <div className="flex justify-between items-start mb-2">
    <h5 className="font-bold text-gray-900 dark:text-white m-0 leading-none titulo2">
      Crescimento
    </h5>
  </div>
  <Chart
    options={chartData.options}
    series={chartData.series}
    type="area"
    height={350}
  />
</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="ui three column grid">
                  {[
                    { titulo: 'Clima Predominante', valor: climaComum, img: null, subtitulo: '' },
                    { titulo: 'Taxa de Crescimento', valor: `${taxaCrescimento}%`, img: null, subtitulo: '' },
                    {
                      titulo: 'Classe Predominante',
                      valor: classePredominante,
                      img: resultadosDefeitos.find(r => r.classe_prevista === classePredominante)?.imagem ? `data:image/png;base64,${resultadosDefeitos.find(r => r.classe_prevista === classePredominante).imagem}` : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg",
                      subtitulo: 'Semente de trigo',
                    },
                  ].map((col, idx) => (
                    <div className="column" key={idx}>
                      <div className="ui rounded segment titulo1">
                        <h4 style={{ fontWeight: 'bold', fontSize: '18px', marginBottom: '10px', color: 'gray' }}>
                          {col.titulo}
                        </h4>
                        <h2 className="mes" style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '10px' }}>
                          {col.img && <img src={col.img} alt="" style={{ width: '50px', height: '50px', borderRadius: '50%' }} />}
                          {col.valor}
                        </h2>
                        <h5 className="crescimento" style={{ fontSize: '18px', fontWeight: 'bold' }}>
                          {col.subtitulo}
                        </h5>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="ui grid equal-height">
  <div className="sixteen wide column p-5">


    <div className="ui rounded segment flex flex-col gap-6 justify-center items-center p-5">
  {/* Título centralizado dentro do card */}
  <h3 className="titulo2" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
  Distribuição de Características e Probabilidade de Crescimento
  </h3>

  {/* indicadores circulares */}
  <div className="flex flex-wrap justify-center gap-8">
    {[
      { cor: 'red', label: 'Defeitos', valor: dadosCirculares.defeitos },
      { cor: 'green', label: 'Qualidades', valor: dadosCirculares.intact },
      { cor: 'blue', label: 'Prob. Crescimento', valor: dadosCirculares.crescimento },
    ].map((ind, idx) => (
      <div className="flex flex-col items-center" key={idx}>
        <div className="relative w-40 h-40 flex items-center justify-center">
          <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 36 36">
            <circle cx="18" cy="18" r="16" fill="none" className="stroke-current text-gray-200" strokeWidth="4"></circle>
            <circle
              cx="18"
              cy="18"
              r="16"
              fill="none"
              className={`stroke-current text-${ind.cor}-600`}
              strokeWidth="4"
              strokeDasharray="100"
              strokeDashoffset={100 - ind.valor}
              strokeLinecap="round"
            ></circle>
          </svg>
          <span className={`text-center text-sm font-bold text-${ind.cor}-600 z-10`}>{ind.valor}%</span>
        </div>
        <h2>{ind.label}</h2>
      </div>
    ))}
  </div>
</div>

  </div>
</div>

          </div>

          {/* SEÇÃO: Defeitos */}
          <div id="Defeitos" className="ui segment rounded content-section" style={displayStyle('Defeitos')}>
            <div className="center qrcodetitulo">
              <h6>Lista de Classificação</h6>
            </div>
            <div className="ui stackable grid items-center center" style={{ minHeight: '400px' }}>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 p-5 w-full">
                {resultadosDefeitos.map((resultado) => (
                  <div
                    key={resultado.id}
                    className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col items-center p-4 cursor-pointer"
                    onClick={() => navigate(`/item/${resultado.id}`)}
                  >
                    <img
                      src={resultado.imagem ? `data:image/png;base64,${resultado.imagem}` : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg"}
                      alt={resultado.classe_prevista}
                      className="w-32 h-32 object-cover rounded-full"
                    />
                    <h3 className="font-bold text-gray-800 text-lg mt-2">{resultado.classe_prevista}</h3>
                    <p className="text-gray-600 text-sm text-center">
                      Probabilidade: {(resultado.probabilidade*100).toFixed(2)}% | Cidade: {resultado.cidade} | Condição: {resultado.condicao}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default MeuComponente;
