document.addEventListener("DOMContentLoaded", () => {
  // Diagnóstico: verificar todos os elementos usados
  const elements = {
    sidebar: document.getElementById("sidebar"),
    toggleButton: document.getElementById("sidebar-toggle"),
    fileInput: document.getElementById("fileInput"),
    fileNameInput: document.getElementById("file-name"),
    enviarBtn: document.getElementById("enviarBtn"),
    cidade: document.getElementById("cidade"),
    resultado: document.getElementById("resultado"),
  };

  console.log("Diagnóstico de elementos:", elements);

  // Se algum elemento for null, exibir erro claro no console
  for (const [name, el] of Object.entries(elements)) {
    if (el === null) {
      console.error(`Elemento NÃO encontrado: ${name}`);
    }
  }

  const {
    sidebar,
    toggleButton,
    fileInput,
    fileNameInput,
    enviarBtn
  } = elements;

  // --- Garantir que os elementos principais existem ---
  if (!toggleButton || !sidebar) {
    console.error("toggleButton ou sidebar inexistente. Verifique o ID 'sidebar-toggle' e 'sidebar' no HTML.");
    return;
  }

  toggleButton.addEventListener("click", () => {
    sidebar.classList.toggle("hidden");

    const icon = toggleButton.querySelector("i");
    if (!icon) return;

    if (sidebar.classList.contains("hidden")) {
      icon.classList.remove("left");
      icon.classList.add("right");
    } else {
      icon.classList.remove("right");
      icon.classList.add("left");
    }
  });

  if (fileInput && fileNameInput) {
    fileInput.addEventListener("change", () => {
      const fileName = fileInput.files.length
        ? fileInput.files[0].name
        : "Nenhum arquivo selecionado";
      fileNameInput.value = fileName;
    });
  }

  if (!enviarBtn) {
    console.error("Botão enviarBtn não encontrado.");
    return;
  }

  enviarBtn.addEventListener("click", async () => {
    const cidadeInput = document.getElementById("cidade");
    const cidade = cidadeInput ? cidadeInput.value || "Jaragua do Sul" : "Jaragua do Sul";

    if (!fileInput || !fileInput.files.length) {
      alert("Selecione uma imagem");
      return;
    }

    const usuarioData = localStorage.getItem("usuario_logado");
    if (!usuarioData) {
      alert("Usuário não encontrado no localStorage");
      return;
    }

    const usuario = JSON.parse(usuarioData);
    const usuarioId = usuario.id;

    const formData = new FormData();
    formData.append("arquivo", fileInput.files[0]);
    formData.append("cidade", cidade);
    formData.append("usuario_id", usuarioId);

    try {
      const response = await fetch("http://18.216.31.10:8000/processar_imagem", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Erro ao processar imagem");
      }

      const data = await response.json();

      const resultadoDiv = document.getElementById("resultado");
      if (!resultadoDiv) {
        console.error("Elemento 'resultado' não encontrado.");
        return;
      }

      resultadoDiv.classList.remove("hidden");
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
  });
});
