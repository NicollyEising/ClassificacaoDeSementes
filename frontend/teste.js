document.getElementById("enviarBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("fileInput");
    const cidade = document.getElementById("cidade").value || "Jaragua do Sul";

    if (!fileInput.files.length) {
        alert("Selecione uma imagem");
        return;
    }

    const usuarioData = localStorage.getItem("usuario");
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

        // Exibir resultado
        const resultadoDiv = document.getElementById("resultado");
        resultadoDiv.innerHTML = `
            <p>Classe prevista: ${data.classe_prevista}</p>
            <p>Probabilidade: ${data.probabilidade}</p>
            <p>Cidade: ${data.clima.cidade}</p>
            <p>Temperatura: ${data.clima.temperatura}°C</p>
            <p>Condição: ${data.clima.condicao}</p>
            <p>Chance de chuva: ${data.clima.chance_chuva}%</p>
            <img src="data:image/png;base64,${data.imagem_anotada_base64}" alt="Imagem Anotada" style="max-width:400px;"/>
        `;
    } catch (error) {
        alert(error.message);
    }
});