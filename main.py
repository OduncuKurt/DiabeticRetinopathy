import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

st.title("🌲 Random Forest ve LightGBM Sınıflandırma Uygulaması (Yeni Veri Yapısı)")
st.markdown("""
Bu uygulama, yüklediğiniz veri dosyasından özellik çıkarımı yaparak Random Forest ve LightGBM algoritmaları ile sınıflandırma işlemi gerçekleştirir.
""")

# CSV Yükleme
uploaded_file = st.file_uploader("Yeni Formatlı CSV Dosyasını Yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Veri seti başarıyla yüklendi!")
    st.write("### 📊 Veri Seti Önizleme:")
    st.dataframe(df.head())

    # Yeni formatlı veri için otomatik özellik seçimi
    if 'label' not in df.columns or 'filename' not in df.columns:
        st.error("Veride 'label' ve 'filename' sütunları bulunmalıdır.")
    else:
        feature_columns = [col for col in df.columns if col not in ['filename', 'label']]

        st.sidebar.header("Özellik Seçimi")
        selected_features = st.sidebar.multiselect("Kullanılacak Özellikler", options=feature_columns, default=feature_columns)

        if not selected_features:
            st.error("En az bir özellik seçmelisiniz.")
        else:
            X = df[selected_features]
            y = df['label']

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

            # Özellik Önem Grafiği Karşılaştırması
            st.write("### 📌 Özellik Önem Karşılaştırması")
            importances_rf = rf_model.feature_importances_
            importances_lgb = lgb_model.feature_importances_

            feature_imp_df = pd.DataFrame({
                'Özellik': selected_features,
                'RandomForest': importances_rf,
                'LightGBM': importances_lgb
            })
            feature_imp_df = feature_imp_df.sort_values(by='RandomForest', ascending=False)

            fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
            sns.barplot(x='RandomForest', y='Özellik', data=feature_imp_df, color="skyblue", label="Random Forest", ax=ax_imp)
            sns.barplot(x='LightGBM', y='Özellik', data=feature_imp_df, color="lightgreen", label="LightGBM", alpha=0.6, ax=ax_imp)
            ax_imp.set_title("Özellik Önem Karşılaştırması")
            ax_imp.legend()
            st.pyplot(fig_imp)
