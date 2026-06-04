# AI_TRADE_BOT
This project is an advanced, quantitative trading bot that combines a multi-factor technical analysis engine (like Golden Cross, RSI, MACD) with a machine learning layer built on scikit-learn's Random Forest algorithm. It uses a dynamic "Score-History Table" to track historical win rates of specific technical setups, and then blends this empirical data with the ML model's predictions to calculate an optimal Kelly-criterion position size for each trade. The results of the program are written in mainly hungarian and also it send a notifications to a given discord chanel with webhook.
# HOW TO RUN:
  If you would like to have notifications to you discord server:
    * 1) Create a webhook for your chanel(it needs two chanel, one for bearish and one for bullish movements)
    * 2) Copy its url
    * 3) Find the part in aibotv3.py where yopu have to paste that in(its at start of the code) and paste it
    * 4) Now you can just run the program with: python aibotv3.py
