import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import './index.css';
import './style.css';

function Item() {
  const navigate = useNavigate();

  const { id: itemId } = useParams(); // captura o ID da URL via React Router

  const [sidebarAberto, setSidebarAberto] = useState(true);
  const [itemData, setItemData] = useState(null);
  const [itemImagem, setItemImagem] = useState(null);
  const [carregando, setCarregando] = useState(true);
  const [sidebarHidden, setSidebarHidden] = useState(false);

  const backendURL = "https://api.sementes.lat:8000";
  const imagensURL = "https://api.sementes.lat:5000";

  const toggleSidebar = () => setSidebarAberto(!sidebarAberto);

  // Função para logout
  const handleLogout = () => {
    localStorage.removeItem("usuario_logado");
    sessionStorage.removeItem("usuario_logado");
    navigate("/login", { replace: true });
  };

  // Copiar link para a área de transferência
  const copiarLink = async (e) => {
    e?.preventDefault();
    const url = itemData?.url_detalhes;
    if (!url) {
      alert("Link não disponível.");
      return;
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(url);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = url;
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      alert("Link copiado para a área de transferência.");
    } catch (err) {
      console.error("Erro ao copiar link:", err);
      alert("Não foi possível copiar o link.");
    }
  };

  // Imprimir QR code em nova janela
  const imprimirQRCode = (e) => {
    e?.preventDefault();
    const base64 = itemData?.qrcode_base64;
    if (!base64) {
      alert("QR Code não disponível para impressão.");
      return;
    }
    const dataUrl = `data:image/png;base64,${base64}`;
    const html = `
      <html>
        <head>
          <title>Imprimir QR Code</title>
          <style>
            body, html { margin: 0; padding: 0; height: 100%; display:flex; align-items:center; justify-content:center; }
            img { max-width: 90%; max-height: 90%; }
          </style>
        </head>
        <body>
          <img src="${dataUrl}" alt="QR Code" />
          <script>
            function waitImageThenPrint() {
              const img = document.querySelector('img');
              if (!img) { window.print(); return; }
              if (img.complete) { window.print(); window.onafterprint = function(){ window.close(); }; }
              else {
                img.onload = function(){ window.print(); window.onafterprint = function(){ window.close(); }; };
                img.onerror = function(){ window.print(); window.onafterprint = function(){ window.close(); }; };
              }
            }
            // give the browser a tick to render
            setTimeout(waitImageThenPrint, 100);
          </script>
        </body>
      </html>
    `;
    const w = window.open("", "_blank", "width=600,height=700");
    if (!w) {
      alert("Bloqueador de pop-ups impediu a abertura da janela de impressão.");
      return;
    }
    w.document.open();
    w.document.write(html);
    w.document.close();
  };

  useEffect(() => {
    if (!itemId) {
      console.error("ID do item não informado na URL.");
      setCarregando(false);
      return;
    }

    const carregarItem = async () => {
      try {
        const resposta = await fetch(`${backendURL}/resultado/${itemId}`);
        if (!resposta.ok) {
          setItemData({ erro: "Item não encontrado." });
          setCarregando(false);
          return;
        }
        const item = await resposta.json();

        try {
          const respostaImg = await fetch(`${imagensURL}/resultado/${itemId}`);
          if (respostaImg.ok) {
            const contentType = respostaImg.headers.get("Content-Type");
            if (contentType?.includes("application/json")) {
              const data = await respostaImg.json();
              if (data.imagem) setItemImagem(`data:image/png;base64,${data.imagem}`);
            } else {
              const blob = await respostaImg.blob();
              setItemImagem(URL.createObjectURL(blob));
            }
          } else {
            console.warn("Imagem não encontrada na porta 5000:", respostaImg.status);
          }
        } catch (erroImg) {
          console.warn("Erro ao buscar imagem na porta 5000:", erroImg);
        }

        setItemData(item);
      } catch (erro) {
        console.error("Erro ao carregar item:", erro);
        setItemData({ erro: "Erro ao carregar o item." });
      } finally {
        setCarregando(false);
      }
    };

    carregarItem();
  }, [itemId]);

  if (carregando) return <p className="ui message">Carregando...</p>;
  if (!itemData) return <p className="ui message warning">Nenhum item selecionado.</p>;
  if (itemData.erro) return <p className="ui negative message">{itemData.erro}</p>;

  return (
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
                <Link className="item" to="/modelo"><i className="boxes icon"></i>Modelo Utilizado</Link>
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
        <div id="item-container" className="item" style={{ marginTop: '1rem' }}>
          <div className="ui raised very padded text container segment transition scale-in"
               style={{ maxWidth: '700px', marginTop: '2rem', borderRadius: '1rem' }}>
            <div className="flex flex-col items-center text-center space-y-4">
              <img
                src={itemImagem || "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg"}
                alt="Imagem do Item"
                className="ui circular centered image shadow-lg border-4 border-gray-200"
                style={{ objectFit: 'cover', width: '180px', height: '180px' }}
              />
              <h2 className="ui header text-2xl font-semibold text-gray-800 mt-3">
                {itemData.classe_prevista || "Sem classificação"}
              </h2>
              <div className="w-full text-left bg-gray-50 p-4 rounded-xl mt-4 shadow-sm">
                <p><strong>Cidade:</strong> {itemData.clima?.cidade || "Não informada"}</p>
                <p><strong>Temperatura:</strong> {itemData.clima?.temperatura ?? "N/A"} °C</p>
                <p><strong>Condição:</strong> {itemData.clima?.condicao || "N/A"}</p>
                <p><strong>Chance de Chuva:</strong> {itemData.clima?.chance_chuva ?? "N/A"}%</p>
              </div>
              <div className="text-gray-400 text-sm mt-2">
                <i className="calendar icon"></i>
                {itemData.data_hora ? new Date(itemData.data_hora).toLocaleString("pt-BR") : "Data indisponível"}
              </div>
              {itemData.qrcode_base64 && (
                <div className="flex flex-col items-center mt-4">
                  <img
                    src={`data:image/png;base64,${itemData.qrcode_base64}`}
                    alt="QR Code"
                    className="w-32 h-32 border border-gray-300 rounded-lg shadow-md"
                  />
                </div>
              )}
              {itemData.url_detalhes && (
                <div className="mt-2 text-center" style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
                  <a
                    href={itemData.url_detalhes}
                    target="_blank"
                    rel="noreferrer"
                    className="ui button primary small"
                    onClick={copiarLink}
                    title="Abrir link e copiar"
                  >
                    Copiar Link
                  </a>

                  <button
                    type="button"
                    className="ui button small"
                    onClick={imprimirQRCode}
                    disabled={!itemData.qrcode_base64}
                    title={itemData.qrcode_base64 ? "Imprimir QR Code" : "QR Code não disponível"}
                  >
                    Imprimir QR Code
                  </button>
                </div>
              )}
              {itemData.recomendacoes && itemData.recomendacoes.length > 0 && (
                <div className="ui segment mt-5">
                  <h3 className="ui header">Recomendações</h3>
                  <div className="ui divided items mt-3">
                    {itemData.recomendacoes.map((r, i) => {
                      const acao = r.acao || "Sem ação especificada";
                      const motivo = r.motivo || "";
                      const prioridade = r.prioridade
                        ? <span className={`ui label ${r.prioridade === 'alta' ? 'red' : r.prioridade === 'média' ? 'orange' : 'blue'}`}>{r.prioridade}</span>
                        : null;
                      const score = r.score ? ` (${(r.score * 100).toFixed(1)}%)` : "";
                      const fonte = r.fonte?.titulo ? <p className="text-gray-500 text-sm mt-1"><strong>Fonte:</strong> {r.fonte.titulo}</p> : null;

                      return (
                        <div key={i} className="ui segment text-left mb-3 rounded-xl shadow-sm bg-gray-50">
                          <h4 className="ui header text-lg font-semibold text-gray-700 mb-2">
                            {i + 1}. {acao} {prioridade} {score}
                          </h4>
                          <p className="text-gray-700">{motivo}</p>
                          {fonte}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Item;
