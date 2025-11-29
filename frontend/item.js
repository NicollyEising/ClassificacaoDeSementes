document.addEventListener("DOMContentLoaded", () => {
  'use strict';

  const backendURL = "http://18.216.31.10:8000"; // Porta 8000 para dados
  const imagensURL = "http://18.216.31.10:5000";  // Porta 5000 para imagens

  const urlParams = new URLSearchParams(window.location.search);
  const itemId = urlParams.get("id");
  const container = document.getElementById("item-container");

  if (!itemId) {
    console.error("ID do item não informado na URL.");
    container.innerHTML = "<p class='ui message warning'>Nenhum item selecionado.</p>";
    return;
  }

  async function carregarItem(id) {
    try {
      // Conexão 1: dados do item
      const resposta = await fetch(`${backendURL}/resultado/${id}`);
      if (!resposta.ok) {
        container.innerHTML = "<p class='ui negative message'>Item não encontrado.</p>";
        return;
      }
      const item = await resposta.json();
      console.log("Item recebido:", item);

      // Conexão 2: imagem do item na porta 5000
      let imgHTML = `<div class="ui placeholder segment">
                        <div class="image header">
                          <div class="medium line"></div>
                          <div class="full line"></div>
                        </div>
                     </div>`;

      try {
        const respostaImg = await fetch(`${imagensURL}/resultado/${id}`);
        if (respostaImg.ok) {
          const contentType = respostaImg.headers.get("Content-Type");

          if (contentType && contentType.includes("application/json")) {
            const data = await respostaImg.json();
            if (data.imagem) {
              imgHTML = `<img src="data:image/png;base64,${data.imagem}" 
                               alt="Imagem do Item" 
                               class="ui circular centered image shadow-lg border-4 border-gray-200" 
                               style="object-fit: cover; width: 180px; height: 180px;">`;
            }
          } else {
            const blob = await respostaImg.blob();
            const url = URL.createObjectURL(blob);
            imgHTML = `<img src="${url}" 
                             alt="Imagem do Item" 
                             class="ui circular centered image shadow-lg border-4 border-gray-200" 
                             style="object-fit: cover; width: 180px; height: 180px;">`;
          }
        } else {
          console.warn("Imagem não encontrada na porta 5000:", respostaImg.status);
        }
      } catch (erroImg) {
        console.warn("Erro ao buscar imagem na porta 5000:", erroImg);
      }

      // QR Code
      const qrCodeHTML = item.qrcode_base64
        ? `<div class="flex flex-col items-center mt-4">
             <img src="data:image/png;base64,${item.qrcode_base64}" 
                  alt="QR Code" 
                  class="w-32 h-32 border border-gray-300 rounded-lg shadow-md">
           </div>`
        : "";

      // Link
      const linkHTML = item.url_detalhes
        ? `<div class="mt-2 text-center">
             <a href="${item.url_detalhes}" target="_blank" class="ui button primary small">
               Copiar Link
             </a>
           </div>`
        : "";

      // Recomendações detalhadas
      let recomendacoesHTML = "";
      if (item.recomendacoes && item.recomendacoes.length > 0) {
        const lista = item.recomendacoes.map((r, i) => {
          if (typeof r === "object" && r !== null) {
            const acao = r.acao || "Sem ação especificada";
            const motivo = r.motivo || "";
            const prioridade = r.prioridade ? `<span class="ui label ${r.prioridade === 'alta' ? 'red' : (r.prioridade === 'média' ? 'orange' : 'blue')}">${r.prioridade}</span>` : "";
            const score = r.score ? ` (${(r.score * 100).toFixed(1)}%)` : "";
            const fonte = r.fonte?.titulo ? `<p class="text-gray-500 text-sm mt-1"><strong>Fonte:</strong> ${r.fonte.titulo}</p>` : "";

            return `
              <div class="ui segment text-left mb-3 rounded-xl shadow-sm bg-gray-50">
                <h4 class="ui header text-lg font-semibold text-gray-700 mb-2">
                  ${i + 1}. ${acao} ${prioridade} ${score}
                </h4>
                <p class="text-gray-700">${motivo}</p>
                ${fonte}
              </div>
            `;
          } else {
            return `<div class="ui segment"><p>${r}</p></div>`;
          }
        }).join("");

        recomendacoesHTML = `
          <div class="ui segment mt-5">
            <h3 class="ui header">Recomendações</h3>
            <div class="ui divided items mt-3">
              ${lista}
            </div>
          </div>`;
      }

      // HTML final
      const html = `
        <div class="ui raised very padded text container segment transition scale-in" 
             style="max-width: 700px; margin-top: 2rem; border-radius: 1rem;">
          <div class="flex flex-col items-center text-center space-y-4">
            ${imgHTML}

            <h2 class="ui header text-2xl font-semibold text-gray-800 mt-3">
              ${item.classe_prevista || "Sem classificação"}
            </h2>

            <div class="w-full text-left bg-gray-50 p-4 rounded-xl mt-4 shadow-sm">
              <p><strong>Cidade:</strong> ${item.clima?.cidade || "Não informada"}</p>
              <p><strong>Temperatura:</strong> ${item.clima?.temperatura ?? "N/A"} °C</p>
              <p><strong>Condição:</strong> ${item.clima?.condicao || "N/A"}</p>
              <p><strong>Chance de Chuva:</strong> ${item.clima?.chance_chuva ?? "N/A"}%</p>
            </div>

            <div class="text-gray-400 text-sm mt-2">
              <i class="calendar icon"></i>
              ${item.data_hora ? new Date(item.data_hora).toLocaleString("pt-BR") : "Data indisponível"}
            </div>

            ${qrCodeHTML}
            ${linkHTML}
            ${recomendacoesHTML}
          </div>
        </div>
      `;

      container.innerHTML = html;

    } catch (erro) {
      console.error("Erro ao carregar item:", erro);
      container.innerHTML = "<p class='ui negative message'>Erro ao carregar o item.</p>";
    }
  }

  carregarItem(itemId);
});
