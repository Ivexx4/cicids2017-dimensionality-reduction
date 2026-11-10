# CICIDS2017 PCA

A análise de componentes principais (PCA) do CICIDS2017 está organizada em módulos Python: carregamento e limpeza dos dados, análise, gráficos e apresentação. Os resultados são guardados na pasta `output/`.

## Requisitos e instalação no Windows

1. Instale Python 3.10 ou superior.
2. O ambiente virtual (`.venv`) é local e não é distribuído com o projeto. Crie-o e ative-o no PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Para descarregar automaticamente o dataset, crie um token em Kaggle → Settings → API → Create New Token e guarde o ficheiro `kaggle.json` em `%USERPROFILE%\.kaggle\kaggle.json`. Aceite previamente os termos de acesso ao dataset no Kaggle. Não inclua o token no projeto.
4. Execute a análise:

   ```powershell
   python run_analysis.py
   ```

   Se ainda não existir localmente, o CSV é descarregado e extraído em `data\cicids2017-cleaned-and-preprocessed\`.

Se já tiver o CSV, pode indicar o caminho diretamente, sem autenticação Kaggle:

```powershell
python run_analysis.py --csv "C:\caminho\cicids2017_cleaned.csv"
```

Opções disponíveis:

- `--csv`: caminho para um CSV local.
- `--data-dir`: pasta para procurar/descarregar o dataset.
- `--output-dir`: pasta para guardar os resultados.
- `--chunksize`: número de linhas por bloco durante a leitura.
- `--components`: número de componentes para loadings e scores (predefinição: 14).
- `--variance-threshold`: meta de variância acumulada em percentagem (predefinição: 80).

## Preparação dos dados e análise

O programa remove registos inválidos de acordo com as regras CICFlowMeter aplicadas, converte infinitos em valores em falta, imputa os NaNs pela mediana e remove colunas vazias ou constantes antes da PCA. `Destination Port` é excluída dos preditores e a coluna de classe é usada para identificar as classes nos gráficos, não como entrada da PCA.

A análise inclui o critério de Kaiser (autovalores superiores a 1), o scree plot, loadings e scores até ao número de componentes configurado, com predefinição PC14. O biplot mostra scores e loadings para PC1–PC2.

Para a análise de observações atípicas, o modelo retém as componentes com autovalor superior a 1. Hotelling T² mede a distância no espaço das componentes retidas; Q-residual mede a variação que esse modelo não explica. Os resultados identificam observações para investigação, mas não confirmam, por si só, a existência de ataques. A numeração das observações nos CSV corresponde à ordem dos registos após a limpeza.

## Ficheiros de saída

Em `output/` são gerados:

- `Análise_PCA_CICIDS2017.pptx`, com os gráficos e interpretações;
- gráficos de correlação, critério de Kaiser, variância acumulada, loadings, scores, biplot e outliers;
- `variancia_componentes.csv`, `loadings.csv` e `loadings_dominantes.csv`;
- `top10_hotelling_t2.csv` e `top10_q_residual.csv`, com as dez observações mais extremas e as respetivas classes.
