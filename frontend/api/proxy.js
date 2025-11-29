import fetch from "node-fetch";
import getRawBody from "raw-body";

export const config = { api: { bodyParser: false } };

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).send("Método não permitido");
    return;
  }

  try {
    const buffer = await getRawBody(req);
    const response = await fetch("http://18.216.31.10:8000/processar_imagem", {
      method: "POST",
      body: buffer,
      headers: {
        ...req.headers,
        host: "18.216.31.10",
      },
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("Erro do backend:", text);
      res.status(response.status).send(text);
      return;
    }

    const contentType = response.headers.get("content-type") || "";
    res.status(response.status).setHeader("content-type", contentType);
    const body = await response.buffer();
    res.send(body);

  } catch (error) {
    console.error(error);
    res.status(500).send("Erro ao encaminhar a requisição para o backend");
  }
}
