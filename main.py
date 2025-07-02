import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

st.title("🌲 Random Forest ve LightGBM Sınıflandırma Karşılaştırması")
st.markdown("""
Bu uygulama, görsel tabanlı özelliklerden elde edilen verileri Random Forest ve LightGBM algoritmaları kullanarak sınıflandırır.
Test oranını ayarlayabilir, iki modelin başarısını anlık olarak inceleyebilirsiniz.
""")

# CSV Yükleme
uploaded_file = st.file_uploader("CSV Dosyanızı Yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Veri seti başarıyla yüklendi!")
    st.write("### 📊 Veri Seti Önizleme:")
    st.dataframe(df.head())

    # Özellik ve hedef ayırımı
    st.sidebar.header("Özellik Seçimi")
    default_features = ['feature1_confidence', 'feature2_x', 'feature3_y',
                       'feature4_width', 'feature5_height', 'feature6_area', 'feature7_aspect_ratio']
    selected_features = st.sidebar.multiselect("Kullanılacak Özellikler", options=df.columns.tolist(), default=default_features)

    if not selected_features or 'label' not in df.columns:
        st.error("Lütfen geçerli özellikler seçin ve 'label' sütununun mevcut olduğundan emin olun.")
    else:
        X = df[selected_features]
        y = df['label']

        # Eğitim-test bölmesi
        test_size = st.slider("Test Veri Oranı (%)", min_value=10, max_value=50, value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Random Forest Modeli
        st.info("Random Forest eğitiliyor...")
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        st.success(f"🌲 Random Forest Doğruluk Oranı: {acc_rf:.4f}")

        st.write("### 🔥 Random Forest Karışıklık Matrisi")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        labels = sorted(np.unique(y))
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax_rf)
        ax_rf.set_title("Random Forest Confusion Matrix")
        st.pyplot(fig_rf)

        # LightGBM Modeli
        st.info("LightGBM eğitiliyor...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            max_depth=-1,
            learning_rate=0.03,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        acc_lgb = accuracy_score(y_test, y_pred_lgb)

        st.success(f"⚡ LightGBM Doğruluk Oranı: {acc_lgb:.4f}")

        st.write("### 🔥 LightGBM Karışıklık Matrisi")
        cm_lgb = confusion_matrix(y_test, y_pred_lgb)
        fig_lgb, ax_lgb = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_lgb)
        ax_lgb.set_title("LightGBM Confusion Matrix")
        st.pyplot(fig_lgb)

