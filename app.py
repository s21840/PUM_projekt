


# źródło danych [https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)

import streamlit as st
import pickle
from datetime import datetime
import pandas as pd


filename = "forest_model.sv"
model_tree = pickle.load(open(filename,'rb'))
model_forest = pickle.load(open(filename,'rb'))
base_data = pd.read_csv("card_transactions_400k.csv")

repeat_retailer_d = {0:"Nie",1:"Tak"}
used_chip_d = {0:"Nie",1:"Tak"}
used_pin_d = {0:"Nie",1:"Tak"}
online_order_d = {0:"Nie",1:"Tak"}

def main():

	st.set_page_config(page_title="Fraud detection")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	with overview:
		st.title("Fraud detection")

	with left:
		repeat_retailer_radio = st.radio("Czy transakcja u powtarzającego się dostawcy: ", list(repeat_retailer_d.keys()), format_func=lambda x : repeat_retailer_d[x] )
		used_chip_radio = st.radio("Czy użyto czipu: ", list(used_chip_d.keys()), index=2, format_func= lambda x: used_chip_d[x] )
		used_pin_radio = st.radio("Czy podano pin: ", list(used_pin_d.keys()), format_func=lambda x : used_pin_d[x])
		online_order_radio = st.radio("Czy płatność online: ", list(online_order_d.keys()), format_func=lambda x : online_order_d[x])

	with right:
		distance_home_slider = st.slider("Odległość od miejsca zamieszkania: ", value=1, min_value=int(base_data["distance_from_home"].min()), max_value=int(base_data["distance_from_home"].max()))
		ditance_last_transaction_slider = st.slider("Odległość od ostatniej transakcji: ", min_value=int(base_data["distance_from_last_transaction"].min()), max_value=int(base_data["distance_from_last_transaction"].max()))
		median_purchase_ratio_slider = st.slider("Ratio: ", min_value=int(base_data["ratio_to_median_purchase_price"].min()), max_value=int(base_data["ratio_to_median_purchase_price"].max()))

	data = [[repeat_retailer_radio, used_chip_radio,  used_pin_radio, online_order_radio, distance_home_slider, ditance_last_transaction_slider, median_purchase_ratio_slider]]
	survival = forest_model.predict(data)
	s_confidence = forest_model.predict_proba(data)

	with prediction:
		st.subheader("Czy podejrzewamy falszywą transakcję?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()

