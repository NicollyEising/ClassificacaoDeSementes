document.addEventListener("DOMContentLoaded", () => {
    'use strict';

    const backendURL = "http://127.0.0.1:5000";
    const usuarioLogado = JSON.parse(localStorage.getItem("usuario_logado"));
    if (!usuarioLogado || !usuarioLogado.id) {
        console.error("Usuário não logado.");
        return;
    }
    const usuarioId = usuarioLogado.id;

    $(document).on('click', '.todosD', function(e) {
        e.preventDefault();
        const section = $(this).data('section') || 'Defeitos';
        $('.content-section').hide();
        $('#' + section).show();
        // opcional: rolar para a seção
        $('html, body').animate({ scrollTop: $('#' + section).offset().top }, 500);
    });


    // Funções de cálculo
    function probabilidadeCrescimento(registro) {
        let pSemente = registro.classe_prevista === "Intact soybeans" ? 0.9 :
                        registro.classe_prevista === "Skin-damaged soybeans" ? 0.6 : 0.5;

        let pCondicao = registro.condicao === "Sol" ? 0.95 :
                        registro.condicao === "Nublado" ? 0.8 :
                        registro.condicao === "Chuva" ? 0.85 : 0.8;

        let pChuva = registro.chance_chuva > 0 ? 1 : 0.8;

        return pSemente * pCondicao * pChuva;
    }

    function taxaGeralCrescimento(registros) {
        if (!registros || registros.length === 0) return 0;
        const soma = registros.reduce((acc, r) => acc + probabilidadeCrescimento(r), 0);
        return (soma / registros.length * 100).toFixed(2);
    }

    function climaMaisFrequente(registros) {
        const contagem = {};
        registros.forEach(r => contagem[r.condicao] = (contagem[r.condicao] || 0) + 1);
        return Object.entries(contagem).reduce((a,b) => b[1] > a[1] ? b : a, ["",0])[0];
    }

    function classeMaisPredominante(registros) {
        const contagem = {};
        registros.forEach(r => contagem[r.classe_prevista] = (contagem[r.classe_prevista] || 0) + 1);
        return Object.entries(contagem).reduce((a,b) => b[1] > a[1] ? b : a, ["",0])[0];
    }

    // Funções para popular listas
    function criarListaItem(resultado) {
        const divLista = document.createElement("div");
        divLista.className = "lista";

        const img = document.createElement("img");
        img.src = resultado.imagem
            ? `data:image/png;base64,${resultado.imagem}`
            : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg";
        img.alt = resultado.classe_prevista;

        const divContent = document.createElement("div");
        divContent.className = "content";

        const aHeader = document.createElement("a");
        aHeader.className = "header";
        aHeader.textContent = resultado.classe_prevista;

        const divDesc = document.createElement("div");
        divDesc.className = "description";
        divDesc.textContent = `Probabilidade: ${(resultado.probabilidade * 100).toFixed(2)}% | Cidade: ${resultado.cidade} | Condição: ${resultado.condicao} | Id: ${resultado.id}`;

        divContent.appendChild(aHeader);
        divContent.appendChild(divDesc);
        divLista.appendChild(img);
        divLista.appendChild(divContent);

        return divLista;
    }

    function popularDefeitos(resultados, classeFiltrar = null) {
        const listaContainer = document.querySelector("#rails .ui.relaxed.divided.list");
        if (!listaContainer) return;
        listaContainer.innerHTML = "";
    
        const filtrados = classeFiltrar 
            ? resultados.filter(r => r.classe_prevista === classeFiltrar)
            : resultados;
    
        filtrados.forEach(resultado => {
            const item = criarListaItem(resultado);
    
            // Adiciona redirecionamento ao clicar no item
            item.addEventListener("click", () => {
                window.location.href = `http://127.0.0.1:5000/frontend/item.html?id=${resultado.id}`;
            });
    
            listaContainer.appendChild(item);
        });
    }

    function popularDropdownClasses(resultados) {
        const dropdownMenu = document.getElementById("filtroClasses");
        if (!dropdownMenu) return;
    
        // Obter classes únicas
        const classesUnicas = [...new Set(resultados.map(r => r.classe_prevista))];
    
        dropdownMenu.innerHTML = ""; // limpar itens existentes
        classesUnicas.forEach(classe => {
            const item = document.createElement("div");
            item.className = "item";
            item.textContent = classe;
    
            // Adiciona evento de clique para filtrar
            item.addEventListener("click", () => {
                popularDefeitos(resultados, classe);
            });
    
            dropdownMenu.appendChild(item);
        });
    }


    function popularRecomendacoes(resultados) {
        const recomContainer = document.querySelector("#Recomendações .ui.relaxed.divided.list");
        if (!recomContainer) return;
        recomContainer.innerHTML = "";
        resultados.forEach(resultado => {
            const divLista = document.createElement("div");
            divLista.className = "lista";

            const img = document.createElement("img");
            img.src = resultado.imagem
                ? `data:image/png;base64,${resultado.imagem}`
                : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg";
            img.alt = resultado.classe_prevista;

            const divContent = document.createElement("div");
            divContent.className = "content";

            const aHeader = document.createElement("a");
            aHeader.className = "header";
            aHeader.textContent = `Recomenda-se tratar: ${resultado.classe_prevista}`;

            divContent.appendChild(aHeader);
            divLista.appendChild(img);
            divLista.appendChild(divContent);
            recomContainer.appendChild(divLista);
        });
    }

    // Função para calcular dados dos gráficos circulares
    function calcularDadosCirculares(resultados) {
        const total = resultados.length;
        if (total === 0) return { defeitos: 0, intact: 0, crescimento: 0 };

        const intactCount = resultados.filter(r => r.classe_prevista === "Intact soybeans").length;
        const naoIntactCount = total - intactCount;
        const taxaCrescimento = taxaGeralCrescimento(resultados);

        return {
            defeitos: ((naoIntactCount / total) * 100).toFixed(2),
            intact: ((intactCount / total) * 100).toFixed(2),
            crescimento: taxaCrescimento
        };
    }

    // Fetch do backend
    fetch(`${backendURL}/resultados/${usuarioId}`)
    .then(res => {
        if (!res.ok) throw new Error("Nenhum resultado encontrado para este usuário");
        return res.json();
    })
    .then(resultados => {
        // Pega no máximo os 5 últimos resultados
        const ultimosResultados = resultados.slice(-9);
        

        $('.ui.dropdown').dropdown();

        // Popula listas apenas com os últimos 5
        popularDefeitos(ultimosResultados);
        popularRecomendacoes(ultimosResultados);
        popularDropdownClasses(ultimosResultados);

        // Calcula métricas com todos os resultados (ou apenas os 5 últimos, se desejar)
        const taxaGeral = taxaGeralCrescimento(ultimosResultados);
        const climaComum = climaMaisFrequente(ultimosResultados);
        const classePredominante = classeMaisPredominante(ultimosResultados);

        // Atualiza HTML
        document.querySelector(".crescimento").textContent = taxaGeral + " de crescimento";
        document.querySelector(".mes").textContent = climaComum;
        document.querySelectorAll(".crescimento")[1].textContent = classePredominante;

        // Atualizar gráfico de área
        if (window.ApexCharts) {
            const options = {
                chart: { type: 'area', height: 350 },
                series: [{
                    name: 'Probabilidade de Crescimento',
                    data: ultimosResultados.map(r => (probabilidadeCrescimento(r)*100).toFixed(2))
                }],
                xaxis: { categories: ultimosResultados.map(r => new Date(r.data_hora).toLocaleDateString()) },
                colors: ['rgb(55, 208, 75)'],
                fill: { type: 'solid', opacity: 0.3, colors: ['rgb(55, 208, 75)'] }
            };
            const chart = new ApexCharts(document.querySelector("#area-chart"), options);
            chart.render();
        }

        const dropdown = document.querySelector(".ui.inline.dropdown .menu");
        if (dropdown) {
            dropdown.querySelectorAll(".item").forEach(item => {
                item.addEventListener("click", () => {
                    const classeSelecionada = item.textContent.trim();
                    popularDefeitos(ultimosResultados, classeSelecionada);
                });
            });
        }

        // Atualizar gráficos circulares
        const dadosCirculares = calcularDadosCirculares(ultimosResultados);

        

        // Defeitos
        document.querySelectorAll('.center')[0].querySelector('circle:nth-child(2)').setAttribute('stroke-dasharray', '100');
        document.querySelectorAll('.center')[0].querySelector('circle:nth-child(2)').setAttribute('stroke-dashoffset', 100 - dadosCirculares.defeitos);
        document.querySelectorAll('.center')[0].querySelector('span').textContent = dadosCirculares.defeitos + "%";

        // Qualidades
        document.querySelectorAll('.center')[1].querySelector('circle:nth-child(2)').setAttribute('stroke-dasharray', '100');
        document.querySelectorAll('.center')[1].querySelector('circle:nth-child(2)').setAttribute('stroke-dashoffset', 100 - dadosCirculares.intact);
        document.querySelectorAll('.center')[1].querySelector('span').textContent = dadosCirculares.intact + "%";

        // Probabilidade de Crescimento
        document.querySelectorAll('.center')[2].querySelector('circle:nth-child(2)').setAttribute('stroke-dasharray', '100');
        document.querySelectorAll('.center')[2].querySelector('circle:nth-child(2)').setAttribute('stroke-dashoffset', 100 - dadosCirculares.crescimento);
        document.querySelectorAll('.center')[2].querySelector('span').textContent = dadosCirculares.crescimento + "%";



        const containerDefeitos = document.querySelector("#Defeitos .ui.stackable.grid");
        if (containerDefeitos) {
        const listaCompleta = document.createElement("div");
        listaCompleta.className = "grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 p-5 w-full";

        resultados.forEach(resultado => {
            const card = document.createElement("div");
            card.className = "bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300";

            card.addEventListener("click", () => {
                window.location.href = `http://127.0.0.1:5500/frontend/item.html?id=${resultado.id}`;
            });

            const img = document.createElement("img");
            img.src = resultado.imagem
                ? `data:image/png;base64,${resultado.imagem}`
                : "https://soystats.com/wp-content/uploads/single-soybean-1024x851.jpg";
            img.alt = resultado.classe_prevista;
            img.className = "w-32 h-32 mx-auto object-cover rounded-full mt-4";

            const content = document.createElement("div");
            content.className = "p-4 bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300 flex flex-col items-center";

            const titulo = document.createElement("h3");
            titulo.className = "font-bold text-gray-800 text-lg mb-2";
            titulo.textContent = resultado.classe_prevista;

            const descricao = document.createElement("p");
            descricao.className = "text-gray-600 text-sm";
            descricao.textContent = `Probabilidade: ${(resultado.probabilidade * 100).toFixed(2)}% | Cidade: ${resultado.cidade} | Condição: ${resultado.condicao} | Id: ${resultado.id}`;

            content.appendChild(titulo);
            content.appendChild(descricao);

            card.appendChild(img);
            card.appendChild(content);

            listaCompleta.appendChild(card);


            
});

containerDefeitos.insertBefore(listaCompleta, containerDefeitos.firstChild);

}
    })
    .catch(err => console.error(err));
});
