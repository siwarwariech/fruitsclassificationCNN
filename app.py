import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go




from tensorflow.keras.models import load_model
import keras.backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
def spatial_avg(x):
    import tensorflow as tf
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable()
def spatial_max(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

# Charger les modèles
model_cnn = load_model("model/cnn_model.h5")
model_resnet = load_model("model/resnet_model.h5")
model_cnn_cmb = load_model("model/cnn_cmb_model.h5")


# Charger les données
df = pd.read_csv("data/fruits_dataset.csv")

# Liste des classes
classes = ['fresh_peaches_done', 'fresh_pomegranates_done', 'fresh_strawberries_done',
           'rotten_peaches_done', 'rotten_pomegranates_done', 'rotten_strawberries_done']

# Initialiser l'application
app = dash.Dash(__name__,suppress_callback_exceptions=True ,external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
])

app.title = "Fruit Classifier Dashboard"
server = app.server

# Fonction utilitaire
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{encoded}'

def layout_image_preview():
    return html.Div([
        html.H3("Image Preview by Class"),
        html.Label("Choose a class:"),
        dcc.Dropdown(
            id='preview-class-dropdown',
            options=[{'label': cls, 'value': cls} for cls in classes],
            value=classes[0]
        ),
        html.Div(id='preview-images-output')  # ID distinct ici
    ])



# Layout: Distribution des classes
import plotly.express as px
from dash import html, dcc

def layout_class_distribution():
    class_counts = df['label'].value_counts().reset_index()
    class_counts.columns = ['label', 'count']

    fig = px.pie(
        class_counts,
        names='label',
        values='count',
        title="Répartition des Classes",
        hole=0.3  # facultatif : pour un effet donut
    )
    fig.update_traces(textinfo='percent+label')

    return html.Div([
        html.H3(" Class Distribution", style={'textAlign': 'center'}),
        dcc.Graph(figure=fig)
    ])


# Layout: DataFrame head
def layout_dataframe():
    preview_df = df.head(10).copy()
    preview_df['image'] = preview_df['image'].apply(lambda x: str(x)[:100] + '...' if len(str(x)) > 100 else str(x))
    return html.Div([
        html.H3("Raw Data Head"),
        dbc.Table.from_dataframe(preview_df, striped=True, bordered=True, hover=True)
    ])

# Layout: Comparaison des modèles
def layout_model_compare():
    return html.Div([
        html.H3("Comparer les modèles"),
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('select an image')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
            accept='image/*'
        ),
        html.Div(id='output-image-upload')

    ])

# Layout: Interprétation
def layout_interpretation():
    return html.Div([
        html.H3("Model Interpretation"),
        html.P("Visualisation des couches internes et attention maps à venir.")
    ])



def update_output(content, filename):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded)).resize((128, 128))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds_cnn = model_cnn.predict(img_array)
        preds_resnet = model_resnet.predict(img_array)
        preds_cmb = model_cnn_cmb.predict(img_array)

        label_cnn = classes[np.argmax(preds_cnn)]
        label_resnet = classes[np.argmax(preds_resnet)]
        label_cmb = classes[np.argmax(preds_cmb)]

        return html.Div([
            html.H4("Prediction Results", style={'textAlign': 'center', 'fontWeight': 'bold'}),

            html.Div([
                html.Div([
                    html.P("CNN", style={'fontWeight': 'bold'}),
                    html.P(label_cnn, style={'color': '#28a745'})
                ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

                html.Div([
                    html.P("ResNet", style={'fontWeight': 'bold'}),
                    html.P(label_resnet, style={'color': '#6f42c1'})
                ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),

                html.Div([
                    html.P("CNN+CMB", style={'fontWeight': 'bold'}),
                    html.P(label_cmb, style={'color': '#e83e8c'})
                ], style={'width': '30%', 'display': 'inline-block', 'textAlign': 'center'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),

            html.Div([
                html.Img(src=content, style={'height': '300px', 'borderRadius': '10px'})
            ], style={'textAlign': 'center'})
        ], style={
            'border': '1px solid #ddd',
            'borderRadius': '10px',
            'padding': '20px',
            'marginTop': '30px',
            'backgroundColor': '#ffffff',
            'boxShadow': '0 0 10px rgba(0,0,0,0.05)'
        })

    return html.Div()

# Sidebar
sidebar = html.Div([
    html.Div([
         html.Img(src='/assets/logo.PNG', style={'width': '60px', 'marginRight': '10px'}),
        html.H2("Fruit Quality", style={'color': 'pink', 'margin': 0})
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.Hr(),

    html.Div([
    dcc.Link('Image Preview', href='/preview', style={'display': 'block', 'padding': '10px'}),
    dcc.Link('Class Distribution', href='/distribution', style={'display': 'block', 'padding': '10px'}),
    dcc.Link('DataFrame', href='/dataframe', style={'display': 'block', 'padding': '10px'}),
    dcc.Link('Model Visuals', href='/model-visuals', style={'display': 'block', 'padding': '10px'}),
    dcc.Link('Comparison', href='/compare', style={'display': 'block', 'padding': '10px'}),
    dcc.Link('Interpretation', href='/interpret', style={'display': 'block', 'padding': '10px'})
])

], style={
    'backgroundColor': '#1a1a1a',
    'width': '250px',
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'padding': '20px',
    'color': 'white'
})

# Main content
content = html.Div(id="page-content", style={"marginLeft": "260px", "padding": "2rem 1rem"})

# App Layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# Routing callback
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
            html.H1(" Welcome to My Dashboard", style={'textAlign': 'center', 'marginTop': '100px'}),
            html.P("Explore the sections from the sidebar to get started.",
                   style={'textAlign': 'center', 'fontSize': '18px'})
        ])
    
    elif pathname == "/preview":
        return layout_image_preview()
    elif pathname == "/distribution":
        return layout_class_distribution()
    elif pathname == "/dataframe":
        return layout_dataframe()
    elif pathname == "/model-visuals":
        return layout_model_visuals()
    elif pathname == "/compare":
        return layout_model_compare()
    elif pathname == "/interpret":
        return layout_interpretation()
    else:
        return html.H1("404 - Page not found", style={'textAlign': 'center'})
    # App Layout
app.layout = html.Div([
       dcc.Location(id="url"),
       sidebar,
       html.Div(id="page-content", style={
           "marginLeft": "260px",
           "padding": "2rem 1rem",
           "minHeight": "100vh"
      })
])


@app.callback(
    Output('preview-images-output', 'children'),
    Input('preview-class-dropdown', 'value')
)
def update_preview_images(selected_class):
    folder = os.path.join("images", selected_class)
    if not os.path.exists(folder):
        return html.P("❌ No images found for this class.")

    images = os.listdir(folder)[:20]  # Limite à 8 images
    return html.Div([
        html.Div([
            html.Img(src=encode_image(os.path.join(folder, img)),
                     style={'height': '120px', 'margin': '5px'}),
            html.P(img, style={'fontSize': '12px'})
        ], style={'display': 'inline-block', 'textAlign': 'center'})
        for img in images
    ])
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

def layout_model_visuals():
    # === Chargement ===
    y_true = np.load("y_test_enc.npy")
    y_pred_cnn = np.load("y_pred_cnn.npy")
    y_pred_resnet = np.load("y_pred_resnet.npy")
    y_pred_cnn_cmb = np.load("y_pred_cnn_cmb.npy")

    # Uniformiser taille
    min_len = min(len(y_true), len(y_pred_cnn), len(y_pred_resnet), len(y_pred_cnn_cmb))
    y_true = y_true[:min_len]
    y_pred_cnn = y_pred_cnn[:min_len]
    y_pred_resnet = y_pred_resnet[:min_len]
    y_pred_cnn_cmb = y_pred_cnn_cmb[:min_len]

    # === MATRICES DE CONFUSION ===
    fig_cm_cnn = ff.create_annotated_heatmap(
        z=confusion_matrix(y_true, y_pred_cnn),
        x=classes,
        y=classes,
        colorscale='Blues'
)
    fig_cm_cnn.update_layout(title="Confusion Matrix - CNN", xaxis_title="Predicted", yaxis_title="Actual")

    fig_cm_resnet = ff.create_annotated_heatmap(
        z=confusion_matrix(y_true, y_pred_resnet),
        x=classes,
        y=classes,
        colorscale='Purples'
)
    fig_cm_resnet.update_layout(title="Confusion Matrix - ResNet", xaxis_title="Predicted", yaxis_title="Actual")

    fig_cm_cnn_cmb = ff.create_annotated_heatmap(
        z=confusion_matrix(y_true, y_pred_cnn_cmb),
        x=classes,
        y=classes,
        colorscale='Reds'
)
    fig_cm_cnn_cmb.update_layout(title="Confusion Matrix - CNN+CMB", xaxis_title="Predicted", yaxis_title="Actual")

    # === ROC pour classe 1 uniquement ===
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true == 1, y_pred_cnn == 1)
    fpr_resnet, tpr_resnet, _ = roc_curve(y_true == 1, y_pred_resnet == 1)
    fpr_cmb, tpr_cmb, _ = roc_curve(y_true == 1, y_pred_cnn_cmb == 1)

    auc_cnn = auc(fpr_cnn, tpr_cnn)
    auc_resnet = auc(fpr_resnet, tpr_resnet)
    auc_cmb = auc(fpr_cmb, tpr_cmb)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_cnn, y=tpr_cnn, mode='lines', name=f'CNN (AUC={auc_cnn:.2f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_resnet, y=tpr_resnet, mode='lines', name=f'ResNet (AUC={auc_resnet:.2f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_cmb, y=tpr_cmb, mode='lines', name=f'CNN+CMB (AUC={auc_cmb:.2f})'))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")

    # === F1 Score ===
    f1_cnn = f1_score(y_true, y_pred_cnn, average='weighted')
    f1_resnet = f1_score(y_true, y_pred_resnet, average='weighted')
    f1_cmb = f1_score(y_true, y_pred_cnn_cmb, average='weighted')

    fig_f1 = px.bar(
        x=["CNN", "ResNet", "CNN+CMB"],
        y=[f1_cnn, f1_resnet, f1_cmb],
        title="F1 Score Comparison",
        labels={"x": "Model", "y": "F1 Score"}
    )

    return html.Div([
        html.H3("Model Visualizations (Confusion Matrix, ROC, F1 Score)"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cm_cnn), width=4),
            dbc.Col(dcc.Graph(figure=fig_cm_resnet), width=4),
            dbc.Col(dcc.Graph(figure=fig_cm_cnn_cmb), width=4),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_roc), width=6),
            dbc.Col(dcc.Graph(figure=fig_f1), width=6),
        ])
    ])

def layout_model_compare():
    return html.Div([
        html.H3("Compare Models", style={
            'textAlign': 'center',
            'marginBottom': '30px',
            'color': '#333',
            'fontWeight': 'bold'
        }),

        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.I(className="fas fa-upload", style={'fontSize': '36px', 'color': '#ff99cc'}),
                    html.Br(),
                    html.Span('Drag & Drop or ', style={'fontSize': '16px'}),
                    html.A('Browse Files', style={'fontWeight': 'bold', 'fontSize': '16px', 'color': '#ff66aa'})
                ]),
                style={
                    'width': '100%',
                    'height': '180px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '15px',
                    'textAlign': 'center',
                    'backgroundColor': '#fff',
                    'color': '#333',
                    'cursor': 'pointer',
                    'padding': '20px',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
                },
                accept='image/*',
                multiple=False
            )
        ], style={'maxWidth': '600px', 'margin': '0 auto'}),

        html.Div(id='output-image-upload', style={
            'marginTop': '40px',
            'textAlign': 'center'
        })
    ])

def layout_interpretation():
    # Charger les prédictions sauvegardées
    y_true = np.load("y_test_enc.npy")
    y_pred_cnn = np.load("y_pred_cnn.npy")
    y_pred_resnet = np.load("y_pred_resnet.npy")
    y_pred_cnn_cmb = np.load("y_pred_cnn_cmb.npy")

    
    # Convertir y_true depuis one-hot vers labels entiers
    y_true = y_true.astype(int)  # Juste pour être sûr que ce sont bien des entiersy_true = np.argmax(y_true, axis=1)
    # Calcul des erreurs
    error_cnn = np.mean(y_pred_cnn != y_true)
    error_resnet = np.mean(y_pred_resnet != y_true)
    error_cmb = np.mean(y_pred_cnn_cmb != y_true)

# Détermination du meilleur modèle
    errors = {
        "CNN": error_cnn,
        "ResNet": error_resnet,
        "CNN+CMB": error_cmb
}
    best_model = min(errors, key=errors.get)

# Texte formaté
    text = f"""
        Model Results:

        🧠 CNN:
          - Error rate: {error_cnn:.2%}
          - Simple model but fast.

        🧠 ResNet:
          - Error rate: {error_resnet:.2%}
          - Deeper, more accurate, but slower.

        🧠 CNN + CBAM:
          - Error rate: {error_cmb:.2%}
          - Combined approach, balances accuracy and speed.

        📊 Comparison:
           👉 The best performing model is: **{best_model}**
"""


    return html.Div([
    html.H3("Model Interpretation"),
    html.Pre(text, style={
        "backgroundColor": "#f9f9f9",
        "padding": "15px",
        "borderRadius": "5px",
        "fontSize": "16px",
        "whiteSpace": "pre-wrap"
    })
])


@app.callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def update_output(content, filename):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded)).resize((128, 128))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prédictions
        preds_cnn = model_cnn.predict(img_array)
        preds_resnet = model_resnet.predict(img_array)
        preds_cmb = model_cnn_cmb.predict(img_array)

        label_cnn = classes[np.argmax(preds_cnn)]
        label_resnet = classes[np.argmax(preds_resnet)]
        label_cmb = classes[np.argmax(preds_cmb)]

        return html.Div([
            html.H4("Model Predictions"),
            html.Div([
                html.Div([
                    html.P("CNN", style={'fontWeight': 'bold'}),
                    html.P(label_cnn, style={'color': 'green'})
                ], style={'display': 'inline-block', 'width': '30%'}),
                html.Div([
                    html.P("ResNet", style={'fontWeight': 'bold'}),
                    html.P(label_resnet, style={'color': 'purple'})
                ], style={'display': 'inline-block', 'width': '30%'}),
                html.Div([
                    html.P("CNN + CBAM", style={'fontWeight': 'bold'}),
                    html.P(label_cmb, style={'color': 'red'})
                ], style={'display': 'inline-block', 'width': '30%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around'}),
            html.Img(src=content, style={'height': '250px', 'marginTop': '20px', 'borderRadius': '8px'})
        ])
    return html.Div("No image uploaded.")
 


app.run(debug=True)

