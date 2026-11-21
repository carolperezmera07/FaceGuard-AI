"""
FaceGuard AI — Aplicación Web en Streamlit para Clasificación Facial (Teachable Machine Keras Model)
Compatible con Keras 2 y Keras 3 (Python 3.10 a 3.14 en Streamlit Community Cloud).
Permite cargar datasets completos (ZIP o múltiples imágenes), clasificar por lotes o con cámara en vivo,
y calcular métricas científicas de evaluación: Precisión, Recall, F1-Score, Matriz de Confusión y Curvas ROC / AUC.
"""

import os
import io
import json
import zipfile
import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Configuración de la página
st.set_page_config(
    page_title="FaceGuard AI | Clasificación Facial Teachable Machine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class CustomDepthwiseConv2D(DepthwiseConv2D):
    """Elimina el argumento legado 'groups=1' exportado por Google Teachable Machine."""
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def _deserialize_functional_config(m1_cfg, custom_objects):
    """Deserializa un diccionario de configuración Functional tanto en Keras 3 como en Keras 2."""
    try:
        from keras.src.legacy.saving import saving_utils
        return saving_utils.model_from_config(m1_cfg, custom_objects=custom_objects)
    except Exception:
        return tf.keras.models.model_from_json(json.dumps(m1_cfg), custom_objects=custom_objects)


def _load_flattened_teachable_machine_h5(model_path):
    """
    Reconstruye el modelo de Teachable Machine (keras_model.h5) aplanando la jerarquía
    Sequential(Sequential(Functional, GAP), Sequential(Dense, Dense)) en un único grafo
    Functional limpio y cargando los 263 tensores de pesos directamente desde HDF5.
    Soluciona el ValueError de Keras 3 (Python 3.12 - 3.14) en Sequential._maybe_rebuild / input_spec.py.
    """
    custom_objs = {"DepthwiseConv2D": CustomDepthwiseConv2D}

    with h5py.File(model_path, "r") as f:
        raw_cfg = f.attrs.get("model_config")
        if isinstance(raw_cfg, bytes):
            raw_cfg = raw_cfg.decode("utf-8")
        model_config = json.loads(raw_cfg)

        # Extraer el backbone MobileNetV2 (Functional 'model1'), GlobalAveragePooling2D y las capas Dense
        seq1_layers = model_config["config"]["layers"][1]["config"]["layers"]
        m1_cfg = json.loads(json.dumps(seq1_layers[1]))  # Copia profunda
        gap_cfg = seq1_layers[2]
        seq3_layers = model_config["config"]["layers"][2]["config"]["layers"]

        last_backbone_layer = m1_cfg["config"]["output_layers"][0][0]

        # Conectar GlobalAveragePooling2D directamente a la salida del backbone (out_relu)
        gap_node = {
            "class_name": "GlobalAveragePooling2D",
            "config": gap_cfg["config"],
            "name": gap_cfg["config"]["name"],
            "inbound_nodes": [[[last_backbone_layer, 0, 0, {}]]]
        }
        m1_cfg["config"]["layers"].append(gap_node)
        prev_name = gap_node["name"]

        # Conectar las capas Dense del cabezal clasificador (dense_Dense1 y dense_Dense2)
        for d_layer in seq3_layers:
            if d_layer["class_name"] == "Dense":
                d_cfg = dict(d_layer["config"])
                d_cfg.pop("batch_input_shape", None)
                d_node = {
                    "class_name": "Dense",
                    "config": d_cfg,
                    "name": d_cfg["name"],
                    "inbound_nodes": [[[prev_name, 0, 0, {}]]]
                }
                m1_cfg["config"]["layers"].append(d_node)
                prev_name = d_node["name"]

        m1_cfg["config"]["output_layers"] = [[prev_name, 0, 0]]

        # Construir el modelo Functional unificado (sin usar Sequential)
        flat_model = _deserialize_functional_config(m1_cfg, custom_objs)

        # Recolectar todos los tensores de pesos almacenados en f['model_weights']
        tensors_by_layer = {}

        def _collect_weights(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split("/")
                if len(parts) >= 2:
                    l_name = parts[-2]
                    p_name = parts[-1].split(":")[0]  # ej. 'kernel:0' -> 'kernel'
                    tensors_by_layer.setdefault(l_name, {})[p_name] = np.asarray(obj)

        f["model_weights"].visititems(_collect_weights)

        # Asignar los pesos exactos a cada capa del modelo
        canonical_order = [
            "kernel", "depthwise_kernel", "gamma", "beta", "bias", "moving_mean", "moving_variance"
        ]
        for layer in flat_model.layers:
            if layer.name in tensors_by_layer and len(layer.weights) > 0:
                p_map = tensors_by_layer[layer.name]
                weights_to_set = []
                for w_var in layer.weights:
                    p_key = w_var.name.split("/")[-1].split(":")[0]
                    if p_key in p_map:
                        weights_to_set.append(p_map[p_key])
                if len(weights_to_set) != len(layer.weights):
                    weights_to_set = [p_map[k] for k in canonical_order if k in p_map]
                if len(weights_to_set) == len(layer.weights):
                    layer.set_weights(weights_to_set)

        # Garantizar tensores únicos de entrada y salida (no listas)
        inp = flat_model.inputs[0] if isinstance(flat_model.inputs, (list, tuple)) else flat_model.input
        out = flat_model.outputs[0] if isinstance(flat_model.outputs, (list, tuple)) else flat_model.output
        clean_model = tf.keras.Model(inputs=inp, outputs=out, name="faceguard_teachable_machine")
        return clean_model


@st.cache_resource(show_spinner="Cargando modelo keras_model.h5 y labels.txt...")
def load_teachable_machine_model(model_path="keras_model.h5", labels_path="labels.txt"):
    """Carga el modelo Keras exportado de Teachable Machine y su archivo de etiquetas."""
    try:
        model = _load_flattened_teachable_machine_h5(model_path)
    except Exception:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
        )

    with open(labels_path, "r", encoding="utf-8") as f:
        raw_labels = [line.strip() for line in f.readlines() if line.strip()]

    clean_labels = []
    for line in raw_labels:
        parts = line.split(" ", 1)
        clean_labels.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else line)

    return model, raw_labels, clean_labels


def predict_batch(model, batch_input):
    """Ejecuta la predicción garantizando un arreglo NumPy 2D de salida."""
    preds = model.predict(batch_input, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    return np.asarray(preds)


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

    if not os.path.exists("keras_model.h5") or not os.path.exists("labels.txt"):
        st.error("No se encontraron `keras_model.h5` o `labels.txt` en la raíz del repositorio.")
        st.stop()

    model, raw_labels, clean_labels = load_teachable_machine_model()

    # Barra lateral
    with st.sidebar:
        st.header("⚙️ Estado del Modelo")
        st.success("Modelo Activo: `keras_model.h5`")
        st.markdown("**Arquitectura:** MobileNetV2 (`alpha=0.35`) + GlobalAveragePooling2D + Dense(100, ReLU) + Dense(3, Softmax)")
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
            use_local_dataset = st.button("⚡ Cargar Dataset de Prueba del Repositorio")

        items_to_process = []

        if use_local_dataset:
            if os.path.exists("dataset/evaluacion"):
                for root, _, files in os.walk("dataset/evaluacion"):
                    for fname in sorted(files):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            fpath = os.path.join(root, fname)
                            with open(fpath, "rb") as f:
                                items_to_process.append((fpath, f.read()))
            elif os.path.exists("dataset_evaluacion.zip"):
                with zipfile.ZipFile("dataset_evaluacion.zip", "r") as z:
                    for zinfo in z.infolist():
                        if not zinfo.is_dir() and zinfo.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            if "__MACOSX" not in zinfo.filename:
                                items_to_process.append((zinfo.filename, z.read(zinfo.filename)))
            else:
                st.warning("Sube el archivo `dataset_evaluacion.zip` o tus propias imágenes en el selector de la izquierda.")

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
            y_scores = []
            thumbs = []

            progress = st.progress(0, text="Ejecutando inferencia con keras_model.h5...")
            total = len(items_to_process)

            for idx, (fname, file_bytes) in enumerate(items_to_process):
                try:
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    batch_input, resized_img = preprocess_pil_image(pil_img)
                    preds = predict_batch(model, batch_input)[0]

                    pred_idx = int(np.argmax(preds))
                    if forced_class == "Detectar automáticamente por nombre de archivo/carpeta":
                        true_idx = infer_ground_truth(fname, clean_labels)
                    else:
                        true_idx = clean_labels.index(forced_class)

                    y_true.append(true_idx)
                    y_pred.append(pred_idx)
                    y_scores.append(preds)
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
                "y_scores": np.array(y_scores),
                "clean_labels": clean_labels
            }

            st.success(f"¡Dataset procesado exitosamente! Total: {len(df_results)} imágenes clasificadas.")

            c1, c2, c3 = st.columns(3)
            acc_val = accuracy_score(y_true, y_pred) * 100
            c1.metric("Imágenes Procesadas", len(df_results))
            c2.metric("Exactitud del Lote (Accuracy)", f"{acc_val:.2f}%")
            c3.metric("Confianza Promedio", f"{df_results['Confianza (%)'].mean():.2f}%")

            st.dataframe(df_results, use_container_width=True)

            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar Tabla de Resultados (CSV)",
                data=csv_bytes,
                file_name="resultados_clasificacion_faceguard.csv",
                mime="text/csv"
            )

            st.markdown("#### 🖼️ Vista Previa de Muestras Clasificadas")
            cols = st.columns(6)
            for i, (t_name, t_img, t_pred, t_conf) in enumerate(thumbs[:12]):
                with cols[i % 6]:
                    st.image(t_img, caption=f"{t_pred} ({t_conf*100:.1f}%)", use_container_width=True)
        else:
            st.info("Sube imágenes o un archivo `.zip` (por ejemplo `dataset_evaluacion.zip`) para comenzar.")

    # =========================================================================
    # PESTAÑA 2: MÉTRICAS DE EVALUACIÓN (PRECISIÓN, RECALL, F1, ROC Y AUC)
    # =========================================================================
    with tab2:
        st.subheader("📊 Evaluación Científica del Modelo: Precisión, Recall, F1-Score, ROC y AUC")

        if "eval_data" not in st.session_state or len(st.session_state["eval_data"]["y_true"]) == 0:
            st.warning("Primero carga un dataset en la **Pestaña 1** para calcular las métricas en vivo.")
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

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy (Exactitud)", f"{acc*100:.2f}%")
            k2.metric("Precision (Macro)", f"{prec_macro*100:.2f}%")
            k3.metric("Recall (Sensibilidad)", f"{rec_macro*100:.2f}%")
            k4.metric("F1-Score (Macro)", f"{f1_macro*100:.2f}%")
            k5.metric("Macro AUC-ROC", f"{macro_auc:.4f}")

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
                preds = predict_batch(model, batch_input)[0]
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
