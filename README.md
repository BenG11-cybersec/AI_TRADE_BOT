# AI_TRADE_BOT
This project is an advanced, quantitative trading bot that combines a multi-factor technical analysis engine (like Golden Cross, RSI, MACD) with a machine learning layer built on scikit-learn's Random Forest algorithm. It uses a dynamic "Score-History Table" to track historical win rates of specific technical setups, and then blends this empirical data with the ML model's predictions to calculate an optimal Kelly-criterion position size for each trade. The results of the program are written in mainly hungarian and it also sends notifications to a given discord chanel through webhook.
# HOW TO RUN:
**1. Clone the repository**
Open your terminal and clone the code to your local machine:
```bash
git clone https://github.com/BenG11-cybersec/AI_TRADE_BOT.git
cd AI_TRADE_BOT
```
**2.Make a venv**
```bash
python -m venv venv
```
You need to activate it everytime you want to use the bot: 
```bash
Linux/Mac: source venv/bin/activate
Win: venv\Scripts\activate
```

**2. Install Dependencies**
Make sure you have Python installed, then install the required packages(you need to make a venv to do this):
```bash
pip install pandas numpy scikit-learn yfinance python-dotenv
```

**3. Create a `.env` file**
Create a new file named exactly `.env` in the root folder of the project.

**4. Add your Discord Webhook URL**
Open the `.env` file with any text editor and paste your Discord Webhook URLs inside it (you need two discord chanels on your server one for bearish movemenst and one for bullish notifications). Use the exact format below (no quotes and no spaces):
```text
bear_url=https://discord.com/api/webhooks/your_webhook_url_here_for_bear_movements_notifications
bull_url=https://discord.com/api/webhooks/your_webhook_url_here_for_bull_movements_notifications
```

**5. Start the Bot**
Once your webhooks are configured, simply run the live trading bot and see how it works for yourself:
```bash
python aibotv3.py
```
However if you would like to see first how the bot performs in real enviroment compared to quant-only and Buy&Hold strategies then run this first and see how it would have performed with the given stocks in the past 7 years:
```bash
python backtest_readonly.py
```

# How to train the bot and customize:
If you would like to train the bot with your own stocks and data you should do the following steps:

**1.Change the ticker symbol**
First change the tickers in the WATCHLIST in aibotv3.py and in backtest.py for the ones you would like follow and you want the ai to be more specific about.

**2.Gather data**
To have datas for bot on these new stocks run the following command:
```bash
python backtest.py
```
This can take quite a long time, even couple of hours with normal hardwer setup, but if its take too long then you can set the random forest features in ai_layer.py lower, but note that in this case the quality of ai can a little more inaccurate.

**3.train the ai**
Type this command to finally train the ai with the gathered datas:
```bash
python ai_layer.py --train
```
And whenever in the future you would like to train the bot with new datas then first type this command:
```bash
python ai_layer.py --clear-data
```
and then repeat the process from the first point("Change the ticker symbol")
**4.Testing**
And finally before running the aibotv3.py you can test the newly trained bot how it performs, with the backtest_readonly.py as it was formerly mentioned
