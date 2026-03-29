import yfinance as yf
import pandas as pd
import sys

def get_basket_data(tickers, start_date, end_date):
    print(f"Downloading data for: {tickers}...")
    
    # Étape 1 : Téléchargement brut
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    if raw_data.empty:
        print("❌ ERREUR : yfinance n'a rien renvoyé. Vérifiez votre connexion ou mettez à jour yfinance.")
        sys.exit(1)
        
    # Étape 2 : Extraction sécurisée des prix
    try:
        data = raw_data['Adj Close']
    except KeyError:
        print("⚠️ 'Adj Close' introuvable dans la réponse de Yahoo. Utilisation de 'Close'.")
        data = raw_data['Close']
        
    print(f"📊 Lignes téléchargées : {len(data)}")
    
    # Étape 3 : Nettoyage et vérification
    clean_data = data.dropna()
    print(f"🧹 Lignes restantes après dropna() : {len(clean_data)}")
    
    if clean_data.empty:
        print("❌ ERREUR : Le dropna() a supprimé toutes les données. Il y a probablement un ticker invalide ou des dates sans cotation.")
        sys.exit(1)
        
    print("✅ Data download complete.")
    return clean_data

if __name__ == "__main__":
    my_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    df = get_basket_data(my_tickers, '2021-01-01', '2026-01-01')
    
    df.to_csv('basket_prices.csv')
    print("\nAperçu des données sauvegardées :")
    print(df.head())