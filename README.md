# AI_TRADE_BOT
This project is an advanced, quantitative trading bot that combines a multi-factor technical analysis engine (like Golden Cross, RSI, MACD) with a machine learning layer built on scikit-learn's Random Forest algorithm. It uses a dynamic "Score-History Table" to track historical win rates of specific technical setups, and then blends this empirical data with the ML model's predictions to calculate an optimal Kelly-criterion position size for each trade. The results of the program are written in mainly hungarian and also it send a notifications to a given discord chanel with webhook.
# HOW TO RUN:
**1. Clone the repository**
Open your terminal and clone the code to your local machine:
```bash
git clone [https://github.com/YOUR_USERNAME/AI_TRADE_BOT.git](https://github.com/YOUR_USERNAME/AI_TRADE_BOT.git)
cd AI_TRADE_BOT
```

**2. Install Dependencies**
Make sure you have Python installed, then install the required packages:
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
