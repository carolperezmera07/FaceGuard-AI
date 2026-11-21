"""
FaceGuard AI — Aplicación Web en Streamlit para Clasificación Facial (Teachable Machine Keras Model)
Permite cargar datasets completos (ZIP o múltiples imágenes), clasificar por lotes o con cámara en vivo,
y calcular métricas científicas de evaluación: Precisión, Recall, F1-Score, Matriz de Confusión y Curvas ROC / AUC.
Lista para desplegar en Streamlit Community Cloud desde GitHub.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Configuración de la página
st.set_page_config(
    page_title="FaceGuard AI | Clasificación Facial Teachable Machine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Compatibilidad universal con DepthwiseConv2D exportado por Google Teachable Machine
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


@st.cache_resource(show_spinner="Cargando modelo keras_model.h5 y labels.txt...")
def load_teachable_machine_model(model_path="keras_model.h5", labels_path="labels.txt"):
    """Carga el modelo Keras exportado de Teachable Machine y su archivo de etiquetas."""
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
    )
    with open(labels_path, "r", encoding="utf-8") as f:
        raw_labels = [line.strip() for line in f.readlines() if line.strip()]

    # Extraer nombres limpios y etiquetas completas (ej. "0 Con mascarilla")
    clean_labels = []
    for line in raw_labels:
        parts = line.split(" ", 1)
        clean_labels.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else line)

    return model, raw_labels, clean_labels


def preprocess_pil_image(pil_img):
    """Preprocesamiento oficial de Teachable Machine: 224x224 RGB normalizado a [-1, 1]."""
    img_rgb = pil_img.convert("RGB")
    img_resized = ImageOps.fit(img_rgb, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img_resized, dtype=np.float32)
    normalized = (img_array / 127.5) - 1.0
    return np.expand_dims(normalized, axis=0), img_resized


def infer_ground_truth(filename_or_path, clean_labels):
    """Infiere la clase real (Ground Truth) a partir del nombre de carpeta o archivo."""
    lower = filename_or_path.lower().replace("_", " ").replace("-", " ")
    for idx, lbl in enumerate(clean_labels):
        if lbl.lower() in lower:
            return idx
    if "inc" in lower or "mal" in lower:
        return 2 if len(clean_labels) > 2 else 0
    if "sin" in lower or "no mask" in lower or "without" in lower:
        return 1 if len(clean_labels) > 1 else 0
    if "con" in lower or "with mask" in lower or "mask" in lower:
        return 0
    return 0


def main():
    st.title("🛡️ FaceGuard AI — Sistema de Clasificación y Análisis Facial de EPP")
    st.markdown(
        "**Modelo Entrenado en Google Teachable Machine (`keras_model.h5`)** • "
        "Verificación de Bioseguridad Facial (`Con mascarilla`, `Sin mascarilla`, `Mascarilla incorrecta`)"
    )

    # Cargar modelo
    if not os.path.exists("keras_model.h5") or not os.path.exists("labels.txt"):
        st.error("No se encontraron `keras_model.h5` o `labels.txt` en la raíz del repositorio.")
        st.stop()

    model, raw_labels, clean_labels = load_teachable_machine_model()

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Estado del Modelo")
        st.success("Modelo Activo: `keras_model.h5`")
        st.markdown("**Arquitectura:** MobileNetV2 (`alpha=0.35`) + Dense(100, ReLU) + Dense(3, Softmax)")
        st.markdown("**Resolución de Entrada:** `224 × 224 × 3`")
        st.markdown("**Clases Detectadas (`labels.txt`):**")
        for r_lbl in raw_labels:
            st.code(r_lbl)

        st.divider()
        st.markdown("### 📌 Instrucciones Rápidas")
        st.info(
            "1. En la **Pestaña 1**, sube un dataset de imágenes o un archivo `.zip` para clasificar por lotes.\n"
            "2. En la **Pestaña 2**, visualiza **Precisión, Recall, F1-Score, Matriz de Confusión y Curvas ROC/AUC**.\n"
            "3. En la **Pestaña 3**, prueba una imagen individual o tu **Cámara Web**."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 1. Cargar Dataset y Resultados por Lotes",
        "📊 2. Métricas: Precisión, Recall, F1 y Curvas ROC/AUC",
        "📸 3. Clasificación Individual / Cámara Web",
        "📘 4. Contexto y Arquitectura del Modelo"
    ])

    # =========================================================================
    # PESTAÑA 1: CARGA DE DATASET POR LOTES (IMÁGENES O ZIP)
    # =========================================================================
    with tab1:
        st.subheader("📂 Carga de Dataset para Clasificación Masiva")
        col_up1, col_up2 = st.columns([2, 1])

        with col_up1:
            uploaded_files = st.file_uploader(
                "Sube múltiples imágenes (PNG, JPG, WEBP) o un archivo .ZIP con carpetas por clase:",
                type=["png", "jpg", "jpeg", "webp", "zip"],
                accept_multiple_files=True
            )

        with col_up2:
            forced_class = st.selectbox(
                "Etiqueta Real (Ground Truth) para evaluación:",
                options=["Detectar automáticamente por nombre de archivo/carpeta"] + clean_labels
            )
            use_local_dataset = st.button("⚡ Cargar Dataset de Prueba del Repositorio (`dataset/evaluacion`)")

        items_to_process = []

        # Si el usuario pulsa el botón de dataset local del repositorio
        if use_local_dataset and os.path.exists("dataset/evaluacion"):
            for root, _, files in os.walk("dataset/evaluacion"):
                for fname in sorted(files):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        fpath = os.path.join(root, fname)
                        with open(fpath, "rb") as f:
                            items_to_process.append((fpath, f.read()))

        # Si el usuario subió archivos o un ZIP
        if uploaded_files:
            for uf in uploaded_files:
                if uf.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(uf.read())) as z:
                        for zinfo in z.infolist():
                            if not zinfo.is_dir() and zinfo.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                                if "__MACOSX" not in zinfo.filename:
                                    items_to_process.append((zinfo.filename, z.read(zinfo.filename)))
                else:
                    items_to_process.append((uf.name, uf.read()))

        if items_to_process:
            records = []
            y_true = []
            y_pred = []
            y_Scores = []
            thumbs = []

            progress = st.progress(0, text="Ejecutando inferencia con keras_model.h5...")
            total = len(items_to_process)

            for idx, (fname, file_bytes) in enumerate(items_to_process):
                try:
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    batch_input, resized_img = preprocess_pil_image(pil_img)
                    preds = model.predict(batch_input, verbose=0)[0]

                    pred_idx = int(np.argmax(preds))
                    if forced_class == "Detectar automáticamente por nombre de archivo/carpeta":
                        true_idx = infer_ground_truth(fname, clean_labels)
                    else:
                        true_idx = clean_labels.index(forced_class)

                    y_true.append(true_idx)
                    y_pred.append(pred_idx)
                    y_Scores.append(preds)
                    thumbs.append((os.path.basename(fname), resized_img, clean_labels[pred_idx], float(preds[pred_idx])))

                    row = {
                        "Archivo": os.path.basename(fname),
                        "Ruta / Carpeta": fname,
                        "Clase Real (Ground Truth)": clean_labels[true_idx],
                        "Clase Predicha": clean_labels[pred_idx],
                        "Confianza (%)": round(float(preds[pred_idx]) * 100, 2),
                        "Correcto": "✔ Sí" if true_idx == pred_idx else "✖ No"
                    }
                    for c_i, c_name in enumerate(clean_labels):
                        row[f"Prob. {c_name} (%)"] = round(float(preds[c_i]) * 100, 2)
                    records.append(row)
                except Exception as e:
                    st.warning(f"No se pudo procesar {fname}: {e}")

                progress.progress((idx + 1) / total, text=f"Procesadas {idx + 1} de {total} imágenes...")

            df_results = pd.DataFrame(records)
            st.session_state["eval_data"] = {
                "df": df_results,
                "y_true": np.array(y_true),
                "y_pred": np.array(y_pred),
                "y_scores": np.array(y_Scores),
                "clean_labels": clean_labels
            }

            st.success(f"¡Dataset procesado exitosamente! Total: {len(df_results)} imágenes clasificadas.")

            # Resumen rápido
            c1, c2, c3 = st.columns(3)
            acc_val = accuracy_score(y_true, y_pred) * 100
            c1.metric("Imágenes Procesadas", len(df_results))
            c2.metric("Exactitud del Lote (Accuracy)", f"{acc_val:.2f}%")
            c3.metric("Confianza Promedio", f"{df_results['Confianza (%)'].mean():.2f}%")

            st.dataframe(df_results, use_container_width=True)

            # Botón de descarga CSV
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar Tabla de Resultados (CSV)",
                data=csv_bytes,
                file_name="resultados_clasificacion_faceguard.csv",
                mime="text/csv"
            )

            # Galería de miniaturas
            st.markdown("#### 🖼️ Vista Previa de Muestras Clasificadas")
            cols = st.columns(6)
            for i, (t_name, t_img, t_pred, t_conf) in enumerate(thumbs[:12]):
                with cols[i % 6]:
                    st.image(t_img, caption=f"{t_pred} ({t_conf*100:.1f}%)", use_container_width=True)
        else:
            st.info("Sube imágenes o un archivo `.zip`, o haz clic en **'⚡ Cargar Dataset de Prueba del Repositorio'** para comenzar.")

    # =========================================================================
    # PESTAÑA 2: MÉTRICAS DE EVALUACIÓN (PRECISIÓN, RECALL, F1, ROC Y AUC)
    # =========================================================================
    with tab2:
        st.subheader("📊 Evaluación Científica del Modelo: Precisión, Recall, F1-Score, ROC y AUC")

        if "eval_data" not in st.session_state or len(st.session_state["eval_data"]["y_true"]) == 0:
            st.warning("Primero carga o selecciona un dataset en la **Pestaña 1** para calcular las métricas en vivo.")
        else:
            ed = st.session_state["eval_data"]
            y_true = ed["y_true"]
            y_pred = ed["y_pred"]
            y_scores = ed["y_scores"]
            n_classes = len(clean_labels)

            acc = accuracy_score(y_true, y_pred)
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )

            # Binarizar para Curvas ROC One-vs-Rest
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

            fpr, tpr, roc_auc = {}, {}, {}
            valid_aucs = []
            for i in range(n_classes):
                if np.sum(y_true_bin[:, i]) > 0 and np.sum(y_true_bin[:, i]) < len(y_true_bin):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    valid_aucs.append(roc_auc[i])
                else:
                    fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), np.nan

            macro_auc = float(np.nanmean(valid_aucs)) if valid_aucs else 0.0

            # Tarjetas KPI
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy (Exactitud)", f"{acc*100:.2f}%")
            k2.metric("Precision (Macro)", f"{prec_macro*100:.2f}%")
            k3.metric("Recall (Sensibilidad)", f"{rec_macro*100:.2f}%")
            k4.metric("F1-Score (Macro)", f"{f1_macro*100:.2f}%")
            k5.metric("Macro AUC-ROC", f"{macro_auc:.4f}")

            # Tabla por clase
            prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(
                y_true, y_pred, labels=list(range(n_classes)), zero_division=0
            )
            df_metrics = pd.DataFrame({
                "Clase": raw_labels,
                "Soporte (N)": sup_c,
                "Precisión (%)": np.round(prec_c * 100, 2),
                "Recall / Sensibilidad (%)": np.round(rec_c * 100, 2),
                "F1-Score (%)": np.round(f1_c * 100, 2),
                "AUC-ROC (OvR)": [round(roc_auc[i], 4) if not np.isnan(roc_auc[i]) else "N/A" for i in range(n_classes)]
            })
            st.markdown("#### 📑 Tabla Desglosada de Métricas por Clase")
            st.dataframe(df_metrics, use_container_width=True)

            # Gráficas lado a lado: Matriz de Confusión y Curvas ROC
            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.markdown("#### 🧩 Matriz de Confusión")
                cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
                fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.5))
                im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.colorbar(im, ax=ax_cm)
                ax_cm.set(
                    xticks=np.arange(n_classes),
                    yticks=np.arange(n_classes),
                    xticklabels=clean_labels,
                    yticklabels=clean_labels,
                    ylabel="Clase Real (Ground Truth)",
                    xlabel="Clase Predicha por el Modelo",
                    title="Matriz de Confusión Multiclase"
                )
                plt.setp(ax_cm.get_xticklabels(), rotation=20, ha="right")
                thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
                for i in range(n_classes):
                    for j in range(n_classes):
                        ax_cm.text(
                            j, i, format(cm[i, j], "d"),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight="bold"
                        )
                fig_cm.tight_layout()
                st.pyplot(fig_cm)

            with g_col2:
                st.markdown("#### 📈 Curvas ROC (One-vs-Rest) y AUC")
                fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4.5))
                colors = ["#10b981", "#2563eb", "#f59e0b", "#8b5cf6"]
                for i in range(n_classes):
                    if not np.isnan(roc_auc[i]):
                        ax_roc.plot(
                            fpr[i], tpr[i],
                            color=colors[i % len(colors)],
                            lw=2.5,
                            label=f"{raw_labels[i]} (AUC = {roc_auc[i]:.3f})"
                        )
                ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, label="Azar (AUC = 0.500)")
                ax_roc.set_xlim([-0.02, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR = 1 - Especificidad)")
                ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
                ax_roc.set_title(f"Curvas ROC Multiclase (Macro AUC = {macro_auc:.3f})")
                ax_roc.legend(loc="lower right")
                ax_roc.grid(alpha=0.25)
                fig_roc.tight_layout()
                st.pyplot(fig_roc)

    # =========================================================================
    # PESTAÑA 3: CLASIFICACIÓN INDIVIDUAL Y CÁMARA WEB
    # =========================================================================
    with tab3:
        st.subheader("📸 Prueba Individual o Captura con Cámara Web")
        c_in, c_out = st.columns(2)

        with c_in:
            mode_input = st.radio("Selecciona fuente de imagen:", ["Subir Fotografía", "Usar Cámara Web"])
            single_img_file = None
            if mode_input == "Subir Fotografía":
                single_img_file = st.file_uploader("Sube una imagen facial:", type=["png", "jpg", "jpeg", "webp"], key="single")
            else:
                single_img_file = st.camera_input("Toma una fotografía para verificar el uso de mascarilla:")

        with c_out:
            if single_img_file is not None:
                pil_img = Image.open(single_img_file)
                batch_input, resized = preprocess_pil_image(pil_img)
                preds = model.predict(batch_input, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                top_label = clean_labels[top_idx]
                top_conf = float(preds[top_idx]) * 100

                st.image(resized, caption="Imagen preprocesada (224×224 px)", width=240)

                if top_idx == 0:
                    st.success(f"✅ **ACCESO AUTORIZADO:** {top_label} ({top_conf:.2f}%)")
                elif top_idx == 2:
                    st.warning(f"⚠️ **ALERTA PREVENTIVA:** {top_label} ({top_conf:.2f}%) — Ajuste su mascarilla sobre nariz y mentón.")
                else:
                    st.error(f"🚨 **ACCESO DENEGADO:** {top_label} ({top_conf:.2f}%) — Uso obligatorio de mascarilla.")

                st.markdown("#### Probabilidades por Clase:")
                for i, r_lbl in enumerate(raw_labels):
                    st.progress(float(preds[i]), text=f"{r_lbl}: {preds[i]*100:.2f}%")

    # =========================================================================
    # PESTAÑA 4: CONTEXTO Y ARQUITECTURA DEL MODELO
    # =========================================================================
    with tab4:
        st.subheader("📘 Contexto del Proyecto y Arquitectura de `keras_model.h5`")
        st.markdown("""
        * **Contexto y Relación con el Proyecto:** Automatización del control de bioseguridad en puntos de acceso de hospitales, laboratorios clínicos y plantas de producción de alimentos.
        * **Área de Aplicación:** Análisis Facial y Reconocimiento de Equipos de Protección Personal (EPP) facial.
        * **Arquitectura Interna de `keras_model.h5`:**
          1. **Capa de Entrada:** `InputLayer` con dimensión `[None, 224, 224, 3]` normalizada en el rango `[-1, 1]`.
          2. **Backbone (Extracción de Características):** `MobileNetV2 (alpha=0.35)` pre-entrenada en ImageNet seguida de `GlobalAveragePooling2D` que genera un vector de `1280` características.
          3. **Cabezal Clasificador (`sequential_3`):**
             * `dense_Dense1`: Capa densa de `100` neuronas con activación `ReLU` (`128,100` parámetros).
             * `dense_Dense2`: Capa de salida de `3` neuronas con activación `Softmax` (`300` parámetros) para las clases `0 Con mascarilla`, `1 Sin mascarilla` y `2 Mascarilla incorrecta`.
        """)


if __name__ == "__main__":
    main()
