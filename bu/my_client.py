from kucoin.client import Market, Trade, User

api_key = "6129943dbc85c200065aa222"
api_secret = "8d15be72-287a-44bf-9e21-a7fa6d67ef86"
api_passphrase = "Inthemix1"

market_client = Market(url="https://api.kucoin.com")
trade_client = Trade(api_key, api_secret, api_passphrase, is_sandbox=True)
user_client = User(api_key, api_secret, api_passphrase, is_sandbox=True)
