# Módulo Especial de Consultoria na Área de Dados com Agentes de IA
# Projeto Prático Para Consultoria na Área de Dados com Agentes de IA
# Deploy de App Para Day Trade Analytics Forex em Tempo Real com Agentes de IA, Groq, DeepSeek e AWS Para Monetização

# Imports
import re
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Carrega o arquivo de variáveis de ambiente
load_dotenv()

########## Analytics ##########

# Dicionário de pares Forex suportados: chave = formato legível, valor = ticker do yFinance
FOREX_TICKERS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/BRL": "USDBRL=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "XAU/USD": "XAUUSD=X",   # Ouro
    "XAG/USD": "XAGUSD=X",   # Prata
}

# Converte o par Forex no formato legível (ex: EUR/USD) para o ticker do yFinance (ex: EURUSD=X)
def dsa_converte_ticker_forex(par):
    par = par.upper().replace(" ", "")
    # Verifica no dicionário de pares suportados
    if par in FOREX_TICKERS:
        return FOREX_TICKERS[par], par
    # Tenta converter automaticamente: remove a barra e adiciona =X
    ticker_yf = par.replace("/", "") + "=X"
    return ticker_yf, par

# Usa o cache de dados do Streamlit com TTL de 60 segundos para dados intraday sempre atualizados
# Define a função que extrai dados históricos de um par Forex com base no ticker e período especificado
@st.cache_data(ttl=60)
def dsa_extrai_dados(ticker, period="1d"):

    # Cria um objeto Ticker do Yahoo Finance para o par Forex especificado
    stock = yf.Ticker(ticker)
    
    # Obtém o histórico de cotações do par com intervalo de 1 minuto (dados intraday)
    hist = stock.history(period=period, interval="1m")
    
    # Reseta o índice do DataFrame para transformar a coluna de data em uma coluna normal
    hist.reset_index(inplace=True)

    # Com interval="1m", o yFinance nomeia a coluna como "Datetime" em vez de "Date"
    # Renomeia para "Date" para manter consistência com os gráficos
    if "Datetime" in hist.columns:
        hist.rename(columns={"Datetime": "Date"}, inplace=True)
    
    # Retorna o DataFrame com os dados históricos do par Forex
    return hist

# Define a função para plotar a cotação do par Forex com base no histórico fornecido
def dsa_plot_stock_price(hist, ticker):
    # Cria um gráfico de linha interativo usando Plotly Express
    # O eixo X representa a data e o eixo Y representa o preço de fechamento do par Forex
    # O título do gráfico inclui o par Forex e o período de análise
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} - Cotação (Hoje - Intraday)", markers=True)
    
    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)

# Define a função para plotar um gráfico de candlestick com base no histórico fornecido
def dsa_plot_candlestick(hist, ticker):

    # Cria um objeto Figure do Plotly para armazenar o gráfico
    fig = go.Figure(

        # Adiciona um gráfico de candlestick com os dados do histórico do par Forex
        data=[go.Candlestick(x=hist['Date'],        # Define as datas no eixo X
                             open=hist['Open'],     # Define os preços de abertura
                             high=hist['High'],     # Define os preços mais altos
                             low=hist['Low'],       # Define os preços mais baixos
                             close=hist['Close'])]  # Define os preços de fechamento
    )
    
    # Atualiza o layout do gráfico, incluindo um título dinâmico com o par Forex
    fig.update_layout(title=f"{ticker} - Candlestick Chart (Hoje - Intraday)")
    
    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)

# Define a função para plotar médias móveis com base no histórico fornecido
def dsa_plot_media_movel(hist, ticker):

    # Calcula a Média Móvel Simples (SMA) de 20 minutos e adiciona ao DataFrame
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    
    # Calcula a Média Móvel Exponencial (EMA) de 20 minutos e adiciona ao DataFrame
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    
    # Cria um gráfico de linha interativo usando Plotly Express
    # Plota a cotação de fechamento, a SMA de 20 períodos e a EMA de 20 períodos
    fig = px.line(hist, 
                  x='Date', 
                  y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} - Médias Móveis (Hoje - Intraday)",  # Define o título do gráfico
                  labels={'value': 'Cotação', 'Date': 'Data'})           # Define os rótulos dos eixos
    
    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)

# Define a função para plotar o volume de negociação do par Forex com base no histórico fornecido
def dsa_plot_volume(hist, ticker):

    # Cria um gráfico de barras interativo usando Plotly Express
    # O eixo X representa a data e o eixo Y representa o volume negociado
    fig = px.bar(hist, 
                 x='Date', 
                 y='Volume', 
                 title=f"{ticker} - Volume de Negociação (Hoje - Intraday)")  # Define o título do gráfico
    
    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig)

########## Agentes de IA ##########

# Agentes de IA 

# Agente de busca na web de forma automatica sobre as paridades de moedas FOREX. (REALIZA BUSCA)
dsa_agente_web_search = Agent(name="DSA Agente Web Search",
                              role="Fazer busca na web",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[DuckDuckGo()],
                              instructions=["Sempre inclua as fontes"],
                              show_tool_calls=True, markdown=True)

# Agente de busca de dados financeiro das paridades de moedas FOREX. (REALIZA ANALISE)
dsa_agente_financeiro = Agent(name="DSA Agente Financeiro",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[YFinanceTools(stock_price=True,
                                                   analyst_recommendations=True,
                                                   stock_fundamentals=True,
                                                   company_news=True)],
                              instructions=["Use tabelas para mostrar os dados",
                                            "Foque em análise de pares de moedas e commodities Forex"],
                              show_tool_calls=True, markdown=True)


multi_ai_agent = Agent(team=[dsa_agente_web_search, dsa_agente_financeiro],
                       model=Groq(id="llama-3.3-70b-versatile"),
                       instructions=["Sempre inclua as fontes", "Use tabelas para mostrar os dados"],
                       show_tool_calls=True, markdown=True)

########## App Web ##########

# Configuração da página do Streamlit
st.set_page_config(page_title="Data Science Academy - Forex Analytics", page_icon=":currency_exchange:", layout="wide")

# Barra Lateral com instruções
st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:

- Insira o par Forex desejado no campo central (ex: `EUR/USD`).
- Clique no botão **Analisar** para obter a análise em tempo real com visualizações e insights gerados por IA.

### Pares Forex suportados:
| Par | Descrição |
|---|---|
| EUR/USD | Euro / Dólar |
| GBP/USD | Libra / Dólar |
| USD/JPY | Dólar / Iene |
| USD/CHF | Dólar / Franco Suíço |
| AUD/USD | Dólar Australiano / Dólar |
| USD/CAD | Dólar / Dólar Canadense |
| NZD/USD | Dólar Neozelandês / Dólar |
| USD/BRL | Dólar / Real Brasileiro |
| EUR/GBP | Euro / Libra |
| EUR/JPY | Euro / Iene |
| XAU/USD | Ouro / Dólar |
| XAG/USD | Prata / Dólar |

### Finalidade da App:
Este aplicativo realiza análises avançadas de pares Forex em tempo real utilizando Agentes de IA com modelo DeepSeek através do Groq e infraestrutura AWS para apoio a estratégias de Day Trade para monetização.
""")

# Botão de suporte na barra lateral
if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: suporte@datascienceacademy.com.br")

# Título principal
st.title(":currency_exchange: Data Science Academy - Forex Analytics")

# Interface principal
st.header("Day Trade Forex Analytics em Tempo Real com Agentes de IA")

# Caixa de texto para input do usuário
par_forex = st.text_input("Digite o Par Forex (ex: EUR/USD, XAU/USD, GBP/USD):").upper().strip()

# Se o usuário pressionar o botão, entramos neste bloco
if st.button("Analisar"):

    # Se temos o par Forex
    if par_forex:

        # Converte o par Forex para o formato do yFinance
        ticker_yf, par_legivel = dsa_converte_ticker_forex(par_forex)

        # Inicia o processamento
        with st.spinner(f"Buscando dados de {par_legivel} em Tempo Real. Aguarde..."):
            
            # Obtém os dados usando o ticker no formato yFinance
            hist = dsa_extrai_dados(ticker_yf)

            # Verifica se os dados foram retornados com sucesso
            if hist.empty:
                st.error(f"Não foi possível obter dados para o par {par_legivel}. Verifique o par informado.")
            else:
                # Renderiza um subtítulo
                st.subheader(f"Análise Gerada Por IA — {par_legivel}")
                
                # Executa o time de Agentes de IA com contexto de Forex
                ai_response = multi_ai_agent.run(
                    f"Analise o par Forex {par_legivel} ({ticker_yf}): "
                    f"resuma o cenário macroeconômico atual, os fatores que influenciam esse par, "
                    f"as últimas notícias relevantes e forneça uma visão sobre a tendência de curto prazo para Day Trade. "
                    f"Use tabelas e dados quando disponíveis."
                )

                # Remove linhas que começam com "Running:"
                # Remove o bloco "Running:" e também linhas "transfer_task_to_finance_ai_agent"
                clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to_finance_ai_agent.*\n?)", "", ai_response.content, flags=re.MULTILINE).strip()

                # Imprime a resposta
                st.markdown(clean_response)

                # Renderiza os gráficos
                st.subheader(f"Visualização dos Dados — {par_legivel}")
                dsa_plot_stock_price(hist, par_legivel)
                dsa_plot_candlestick(hist, par_legivel)
                dsa_plot_media_movel(hist, par_legivel)
                dsa_plot_volume(hist, par_legivel)
    else:
        st.error("Par Forex inválido. Insira um par no formato correto (ex: EUR/USD, XAU/USD).")


# Fim
# Obrigado DSA!

