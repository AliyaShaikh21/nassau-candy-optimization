import streamlit as st
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("Nassau Candy Factory Optimization")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)


    df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed', dayfirst=True)

    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days.abs()
    df['Profit'] = df['Sales'] - df['Cost']

   
    product_factory_map = {
        "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
        "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
        "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
        "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
        "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
        "Laffy Taffy": "Sugar Shack",
        "SweeTARTS": "Sugar Shack",
        "Nerds": "Sugar Shack",
        "Fun Dip": "Sugar Shack",
        "Fizzy Lifting Drinks": "Sugar Shack",
        "Everlasting Gobstopper": "Secret Factory",
        "Hair Toffee": "The Other Factory",
        "Lickable Wallpaper": "Secret Factory",
        "Wonka Gum": "Secret Factory",
        "Kazookles": "The Other Factory"
    }

    df['Factory'] = df['Product Name'].map(product_factory_map)

    le_product = LabelEncoder()
    le_factory = LabelEncoder()
    le_region = LabelEncoder()
    le_ship = LabelEncoder()

    df['Product Name'] = le_product.fit_transform(df['Product Name'])
    df['Factory'] = le_factory.fit_transform(df['Factory'])
    df['Region'] = le_region.fit_transform(df['Region'])
    df['Ship Mode'] = le_ship.fit_transform(df['Ship Mode'])

    X = df[['Product Name', 'Factory', 'Region', 'Ship Mode', 'Units', 'Cost']]
    y = df['Lead Time']

    model = RandomForestRegressor()
    model.fit(X, y)

    factory_coords = {
        "Lot's O' Nuts": (32.88, -111.76),
        "Wicked Choccy's": (32.07, -81.08),
        "Sugar Shack": (48.11, -96.18),
        "Secret Factory": (41.44, -90.56),
        "The Other Factory": (35.11, -89.97)
    }

    def calculate_distance(factory):
        lat, lon = factory_coords[factory]
        return math.sqrt(lat**2 + lon**2)

    results = []

    for p in df['Product Name'].unique():
        for f in df['Factory'].unique():

            temp = df[df['Product Name'] == p].copy()
            temp['Factory'] = f

            pred_lead = model.predict(temp[X.columns]).mean()
            profit = temp['Profit'].mean()
            current_lead = temp['Lead Time'].mean()

            improvement = ((current_lead - pred_lead) / current_lead) * 100

            factory_name = le_factory.inverse_transform([f])[0]

            distance = calculate_distance(factory_name)

            score = (
                0.4 * (1 / (pred_lead + 1)) +
                0.3 * profit +
                0.2 * (improvement / 100) +
                0.1 * (1 / (distance + 1))
            )

            results.append({
                "Product": le_product.inverse_transform([p])[0],
                "Factory": factory_name,
                "Score": score
            })

    results_df = pd.DataFrame(results)

    best = results_df.loc[
        results_df.groupby('Product')['Score'].idxmax()
    ]

    st.subheader("Best Factory Recommendation")
    st.dataframe(best)

    st.subheader("Visualization")
    st.bar_chart(best.set_index('Product')['Score'])