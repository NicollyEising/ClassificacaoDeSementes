import fetch from "node-fetch";
import formidable from "formidable";

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Método não permitido" });
    return;
  }

  const form = new formidable.IncomingForm();

  form.parse(req, async (err, fields, files) => {
    if (err) {
      res.status(500).json({ error: "Erro ao processar o upload" });
      return;
    }

    try {
      const formData = new FormData();

      if (files.arquivo) {
        const file = files.arquivo;
        formData.append("arquivo", await fs.promises.readFile(file.filepath), file.originalFilename);
      }

      for (const key in fields) {
        formData.append(key, fields[key]);
      }

      const response = await fetch("http://18.216.31.10:8000/processar_imagem", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      res.status(response.status).json(data);
    } catch (error) {
      console.error(error);
      res.status(500).json({ error: "Erro ao encaminhar a requisição para o backend" });
    }
  });
}
