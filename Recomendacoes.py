from visaoComputacional import *


responde_agro(final_name)

# -------------------- Recomendações profissionais considerando cidade --------------------
def recomenda_soja_por_cidade(semente_identificada, clima, temp, chance_chuva, cidade):
    """
    Gera recomendações profissionais detalhadas para plantio de soja,
    considerando o tipo de semente, clima, temperatura, chuva e cidade.
    """
    recomendacoes = []

    # Recebe a cidade
    cidade = cidade.lower()

    # Listas de cidades por região
    sul = ["jaragua do sul", "blumenau", "joinville", "curitiba", "porto alegre", "florianópolis"]
    norte = ["manaus", "belém", "rio branco", "boa vista", "macapá", "palmas"]
    leste = ["salvador", "recife", "fortaleza", "joão pessoa", "maceió", "aracaju"]
    oeste = ["cuiabá", "goiânia", "campo grande", "brasília"]

    # Identificação da região
    if cidade in sul:
        regiao = "Sul"
    elif cidade in norte:
        regiao = "Norte"
    elif cidade in leste:
        regiao = "Leste"
    elif cidade in oeste:
        regiao = "Oeste"
    else:
        regiao = "Região não especificada"

    # Tipos de solo típicos por região
    solos_por_regiao = {
        "Sul": "predominantemente argiloso a médio argiloso",
        "Norte": "predominantemente latossolo e podzólico, solos ácidos e arenosos em algumas áreas",
        "Leste": "predominantemente latossolos e solos arenosos, boa drenagem natural",
        "Oeste": "predominantemente solos argilo-arenosos, com variações de terra roxa em áreas férteis"
    }

    # Fertilizantes típicos por região
    fertilizantes_por_regiao = {
        "Sul": [
            "Fósforo (P₂O₅) para corrigir deficiência comum em solos argilosos.",
            "Potássio (K₂O) em cobertura, devido à alta extração da cultura.",
            "Calcário para correção de acidez e fornecimento de cálcio e magnésio."
        ],
        "Norte": [
            "Correção de acidez com calcário é prioridade.",
            "Fósforo em doses mais altas devido à fixação em solos ácidos.",
            "Micronutrientes como Zinco e Boro, essenciais para produtividade."
        ],
        "Leste": [
            "Adubação fosfatada e potássica equilibrada.",
            "Aplicação de Enxofre em solos arenosos.",
            "Uso de matéria orgânica para melhorar retenção de água e nutrientes."
        ],
        "Oeste": [
            "Potássio em maior quantidade devido à lixiviação.",
            "Fósforo em linha de plantio para garantir desenvolvimento inicial.",
            "Adubação com enxofre e micronutrientes como Manganês e Zinco."
        ]
    }

    # Definição do solo
    solo_tipo = solos_por_regiao.get(regiao, "solo local não especificado, avaliar textura antes do plantio")

    # Recomendações por região
    if regiao != "Região não especificada":
        recomendacoes.append(f"No solo típico da região {regiao} ({solo_tipo}), sementes inteiras apresentam boa germinação, mas condições específicas devem ser avaliadas antes do plantio.")
        # Fertilizantes regionais
        recomendacoes.append("Fertilizantes recomendados para a região:")
        for fert in fertilizantes_por_regiao[regiao]:
            recomendacoes.append(f"- {fert}")
    else:
        recomendacoes.append(f"No solo de {cidade.title()}, recomenda-se analisar textura e drenagem antes do plantio.")
        recomendacoes.append("Fertilizantes devem ser definidos a partir de análise de solo local.")

    # Recomendações específicas de semente
    if semente_identificada == "Intact soybeans":
        recomendacoes.append("Semente íntegra: ótima taxa de germinação. Plantar em solo bem preparado e drenado.")
    elif semente_identificada == "Immature soybeans":
        recomendacoes.append("Semente imatura: risco de baixa germinação. Evitar plantio em solos frios e encharcados.")
    elif semente_identificada == "Broken soybeans":
        recomendacoes.append("Sementes quebradas: ajustar densidade de semeadura e considerar mistura com sementes de melhor qualidade.")
    elif semente_identificada == "Skin-damaged soybeans":
        recomendacoes.append("Sementes com casca danificada: alto risco de ataque por fungos, tratar antes do plantio.")
    elif semente_identificada == "Spotted soybeans":
        recomendacoes.append("Sementes com manchas: possível contaminação fúngica, realizar teste de germinação e tratamento pré-plantio.")

    # Condições climáticas
    if temp < 18:
        recomendacoes.append("Solo frio: germinação lenta. Plantar após aquecimento do solo acima de 18°C.")
    elif temp > 30:
        recomendacoes.append("Temperatura elevada: monitorar irrigação para evitar estresse térmico.")
    else:
        recomendacoes.append("Temperatura adequada para germinação.")

    if chance_chuva >= 70:
        recomendacoes.append("Alta probabilidade de chuva: garantir boa drenagem e evitar compactação do solo.")
    elif chance_chuva <= 30:
        recomendacoes.append("Baixa probabilidade de chuva: planejar irrigação suplementar nos primeiros 15 dias.")

    # Condições atuais
    if clima and "erro" not in clima:
        condicao = clima["current"]["condition"]["text"]
        umidade_relativa = clima["current"].get("humidity", None)
        recomendacoes.append(f"Condição atual em {cidade}: {condicao}.")
        if umidade_relativa:
            recomendacoes.append(f"Umidade relativa do ar: {umidade_relativa}%. Monitorar para prevenir doenças foliares.")

    # Boas práticas
    recomendacoes.append("Realizar aração ou gradagem leve se solo estiver compactado.")
    recomendacoes.append("Manter densidade de semeadura adequada à cultivar escolhida.")
    recomendacoes.append("Evitar plantio em áreas encharcadas ou com risco de erosão.")

    print("\n=== Recomendações Profissionais de Soja para", cidade, "===")
    for rec in recomendacoes:
        print("-", rec)
    print("=====================================================\n")

    return recomendacoes


# -------------------- Chamada da função --------------------
recomenda_soja_por_cidade(semente_identificada, clima, temp, chance_chuva, cidade)
