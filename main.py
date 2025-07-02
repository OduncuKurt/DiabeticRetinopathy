import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("🌲 Random Forest Gelişmiş Sınıflandırma Uygulaması")
st.markdown("""
Bu uygulama, görsel tabanlı özelliklerden elde edilen verileri Random Forest algoritması kullanarak sınıflandırır.
Test oranını ayarlayabilir, model başarısını anlık olarak inceleyebilirsiniz.
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

        # Parametre ayarlı model eğitimi
        st.info("Random Forest Eğitiliyor...")
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)

        # Tahmin ve değerlendirme
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model Doğruluk Oranı (Accuracy): {acc:.4f}")

        st.write("### 📄 Sınıflandırma Raporu")
        st.text(classification_report(y_test, y_pred))

        st.write("### 🔥 Karışıklık Matrisi")
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(np.unique(y))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")
        plt.title("Karışıklık Matrisi (Confusion Matrix)")
        st.pyplot(fig)

        # Özellik Önem Grafiği
        st.write("### 📌 Özellik Önem Grafiği")
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({'Özellik': selected_features, 'Önem Skoru': importances})
        feature_imp_df = feature_imp_df.sort_values(by='Önem Skoru', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Önem Skoru', y='Özellik', data=feature_imp_df, ax=ax2)
        ax2.set_title("Özelliklerin Göreli Önemi")
        st.pyplot(fig2)
