import { NextResponse } from 'next/server';
import formidable from 'formidable';
import fs from 'fs';

export const config = {
  api: {
    bodyParser: false, // necessário para receber arquivos via FormData
  },
};

export const POST = async (req) => {
  try {
    const form = formidable({ multiples: false });

    const data = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) return reject(err);
        resolve({ fields, files });
      });
    });

    const { fields, files } = data;
    const arquivo = files.arquivo;
    const cidade = fields.cidade;
    const usuario_id = fields.usuario_id;

    if (!arquivo) {
      return NextResponse.json({ detail: 'Nenhum arquivo enviado' }, { status: 400 });
    }

    // Lê o arquivo em buffer e converte para Base64
    const fileBuffer = fs.readFileSync(arquivo.filepath);
    const imagemBase64 = fileBuffer.toString('base64');

    // Retorna exatamente o que veio do frontend
    return NextResponse.json({
      usuario_id,
      cidade,
      nome_arquivo: arquivo.originalFilename,
      imagem_base64: imagemBase64,
    }, { status: 200 });

  } catch (err) {
    return NextResponse.json({ detail: err.message }, { status: 500 });
  }
};

export const GET = async () => {
  return NextResponse.json({ detail: 'Method Not Allowed' }, { status: 405 });
};
